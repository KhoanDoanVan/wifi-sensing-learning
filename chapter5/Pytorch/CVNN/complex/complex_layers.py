import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "ComplexBatchNorm2d",
    "ComplexConv2d",
    "ComplexDropout",
    "ComplexMaxPool2d",
    "ComplexReLU",
    "ComplexToReal",
]


class ComplexReLU(nn.Module):
    def forward(self, x):
        if not torch.is_complex(x) and x.shape[-1] == 2:
            return torch.stack((F.relu(x[..., 0]), F.relu(x[..., 1])), dim=-1)
        if not torch.is_complex(x):
            return F.relu(x)
        return torch.complex(F.relu(x.real), F.relu(x.imag))


class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            dtype=torch.complex64,
        )

    def forward(self, x):
        return self.conv(x)


class ComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.bn_real = nn.BatchNorm2d(num_features)
        self.bn_imag = nn.BatchNorm2d(num_features)

    def forward(self, x):
        if not torch.is_complex(x):
            return self.bn_real(x)
        real = self.bn_real(x.real)
        imag = self.bn_imag(x.imag)
        return torch.complex(real, imag)


class ComplexDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        if not torch.is_complex(x) and x.shape[-1] == 2:
            mask = torch.ones_like(x[..., 0])
            mask = F.dropout(mask, p=self.p, training=True)
            return torch.stack((x[..., 0] * mask, x[..., 1] * mask), dim=-1)
        if not torch.is_complex(x):
            return F.dropout(x, p=self.p, training=True)

        # Share one dropout mask across real/imag parts to preserve phase.
        mask = torch.ones_like(x.real)
        mask = F.dropout(mask, p=self.p, training=True)
        return torch.complex(x.real * mask, x.imag * mask)


class ComplexToReal(nn.Module):
    def forward(self, x):
        if not torch.is_complex(x) and x.shape[-1] == 2:
            return torch.linalg.vector_norm(x, dim=-1)
        if not torch.is_complex(x):
            return x
        return torch.abs(x)


class ComplexMaxPool2d(nn.Module):
    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        dilation=1,
        ceil_mode=False,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode

    def forward(self, x):
        if not torch.is_complex(x):
            return F.max_pool2d(
                x,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                ceil_mode=self.ceil_mode,
            )

        _, indices = F.max_pool2d(
            torch.abs(x),
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            ceil_mode=self.ceil_mode,
            return_indices=True,
        )

        batch, channels, _, _ = x.shape
        flat = x.reshape(batch, channels, -1)
        pooled = flat.gather(dim=2, index=indices.reshape(batch, channels, -1))
        return pooled.reshape_as(indices)
