import importlib.util
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_COMPLEX_LAYERS_PATH = Path(__file__).resolve().parent / "complex_layers.py"
_COMPLEX_LAYERS_SPEC = importlib.util.spec_from_file_location(
    "_cvnn_complex_layers_for_runtime_module",
    _COMPLEX_LAYERS_PATH,
)
_COMPLEX_LAYERS = importlib.util.module_from_spec(_COMPLEX_LAYERS_SPEC)
exec(
    compile(_COMPLEX_LAYERS_PATH.read_text(encoding="utf-8"), str(_COMPLEX_LAYERS_PATH), "exec"),
    _COMPLEX_LAYERS.__dict__,
)

ComplexReLU = _COMPLEX_LAYERS.ComplexReLU
ComplexToReal = _COMPLEX_LAYERS.ComplexToReal


class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_real = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_imag = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias_real = nn.Parameter(torch.empty(out_features))
            self.bias_imag = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias_real", None)
            self.register_parameter("bias_imag", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_real, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_imag, a=math.sqrt(5))
        if self.bias_real is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias_real, -bound, bound)
            nn.init.uniform_(self.bias_imag, -bound, bound)

    def _forward_parts(self, real, imag):
        out_real = F.linear(real, self.weight_real, self.bias_real) - F.linear(
            imag,
            self.weight_imag,
            None,
        )
        out_imag = F.linear(real, self.weight_imag, self.bias_imag) + F.linear(
            imag,
            self.weight_real,
            None,
        )
        return out_real, out_imag

    def forward(self, x):
        if torch.is_complex(x):
            real, imag = self._forward_parts(x.real, x.imag)
            return torch.complex(real, imag)

        if x.shape[-1] != 2:
            raise ValueError("ComplexLinear expects a complex tensor or a trailing size-2 real/imag axis.")

        real, imag = x.unbind(dim=-1)
        out_real, out_imag = self._forward_parts(real, imag)
        return torch.stack((out_real, out_imag), dim=-1)


class ComplexMLP(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc1 = ComplexLinear(in_features, out_features)
        self.act = ComplexReLU()
        self.fc2 = ComplexLinear(out_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class ComplexTransformerEncoder(nn.Module):
    def __init__(
        self,
        origin_dim,
        key_dim,
        query_dim,
        value_dim,
        hidden_dim,
        norm_shape,
        ffn_input_dim,
        ffn_hidden_dim,
        num_heads,
        num_layers,
        dropout=0.0,
    ):
        super().__init__()
        del query_dim, value_dim, hidden_dim, norm_shape, ffn_input_dim
        self.origin_dim = origin_dim
        self.model_dim = key_dim
        self.input_proj = nn.Linear(origin_dim * 2, key_dim * 2)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=key_dim * 2,
            nhead=num_heads,
            dim_feedforward=max(ffn_hidden_dim * 2, key_dim * 4),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(key_dim * 2, key_dim * 2)

    def forward(self, x):
        if torch.is_complex(x):
            x = torch.stack((x.real, x.imag), dim=-1)
        if x.shape[-1] != 2:
            raise ValueError(
                "ComplexTransformerEncoder expects a complex tensor or a trailing size-2 real/imag axis."
            )

        batch, steps, dim, parts = x.shape
        x = x.reshape(batch, steps, dim * parts)
        x = self.input_proj(x)
        x = self.encoder(x)
        x = self.output_proj(x)
        return x.reshape(batch, steps, self.model_dim, 2)
