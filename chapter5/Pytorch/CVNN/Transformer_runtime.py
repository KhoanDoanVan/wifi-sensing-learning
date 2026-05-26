import importlib.util
from pathlib import Path

import torch
import torch.nn as nn
from einops import rearrange

_COMPLEX_MODULE_PATH = Path(__file__).resolve().parent / "complex" / "complex_module_runtime.py"
_COMPLEX_MODULE_SPEC = importlib.util.spec_from_file_location(
    "_cvnn_complex_module_for_transformer_runtime",
    _COMPLEX_MODULE_PATH,
)
_COMPLEX_MODULE = importlib.util.module_from_spec(_COMPLEX_MODULE_SPEC)
exec(
    compile(_COMPLEX_MODULE_PATH.read_text(encoding="utf-8"), str(_COMPLEX_MODULE_PATH), "exec"),
    _COMPLEX_MODULE.__dict__,
)

ComplexMLP = _COMPLEX_MODULE.ComplexMLP
ComplexToReal = _COMPLEX_MODULE.ComplexToReal
ComplexTransformerEncoder = _COMPLEX_MODULE.ComplexTransformerEncoder


class RF_Transformer(nn.Module):
    def __init__(self, num_classes=6, seglen=8, origin_dim=720, dim=720, heads=8, layers=8, dropout=0.0):
        super().__init__()
        self.num_classes = num_classes
        self.seglen = seglen
        self.encoder_origin_dim = origin_dim
        self.dim = dim
        self.heads = heads
        self.layers = layers
        self.dropout = dropout
        self.encoder = ComplexTransformerEncoder(
            origin_dim=self.encoder_origin_dim,
            key_dim=self.dim,
            query_dim=self.dim,
            value_dim=self.dim,
            hidden_dim=self.dim,
            norm_shape=self.dim,
            ffn_input_dim=self.dim,
            ffn_hidden_dim=self.dim,
            num_heads=self.heads,
            num_layers=self.layers,
            dropout=self.dropout,
        )
        self.c2r = ComplexToReal()
        self.mlp = ComplexMLP(in_features=self.dim, out_features=self.dim)
        self.classifier = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.dim, self.num_classes),
        )

    def forward(self, x):
        x = torch.stack((torch.real(x), torch.imag(x)), dim=-1)
        x = rearrange(x, "b (x s) d i -> b x (s d) i", s=self.seglen)
        class_token = torch.zeros_like(x[:, 0:1])
        x = torch.cat((class_token, x), dim=1)
        x = self.encoder(x)
        x = x[:, 0]
        x = self.mlp(x)
        x = self.c2r(x)
        x = self.classifier(x)
        return x
