import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib.util
from pathlib import Path
from einops import rearrange

_COMPLEX_LAYERS_PATH = Path(__file__).resolve().parent / "complex" / "complex_layers.py"
_COMPLEX_LAYERS_SPEC = importlib.util.spec_from_file_location(
    "_cvnn_complex_layers",
    _COMPLEX_LAYERS_PATH,
)
_COMPLEX_LAYERS = importlib.util.module_from_spec(_COMPLEX_LAYERS_SPEC)
exec(
    compile(_COMPLEX_LAYERS_PATH.read_text(encoding="utf-8"), str(_COMPLEX_LAYERS_PATH), "exec"),
    _COMPLEX_LAYERS.__dict__,
)

ComplexDropout = _COMPLEX_LAYERS.ComplexDropout
ComplexMaxPool2d = _COMPLEX_LAYERS.ComplexMaxPool2d
ComplexReLU = _COMPLEX_LAYERS.ComplexReLU
ComplexToReal = _COMPLEX_LAYERS.ComplexToReal



class AlexNet(nn.Module):
    def __init__(self, num_classes=6):
        super(AlexNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=[51, 4], stride=[18, 1], padding=[0, 0], dtype=torch.complex64)
        self.maxpool1 = ComplexMaxPool2d(kernel_size = [3, 2], stride=[2, 1])
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=[5, 5], stride=[1, 1], padding=[2, 2], dtype=torch.complex64)
        self.maxpool2 = ComplexMaxPool2d(kernel_size = [3, 3], stride=[2, 2])
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dtype=torch.complex64)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dtype=torch.complex64)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dtype=torch.complex64)
        self.maxpool3 = ComplexMaxPool2d(kernel_size = [3, 3], stride=[2, 2])
        self.fc1 = nn.Linear(in_features=6400, out_features=4096, dtype=torch.complex64)

        self.c2r = ComplexToReal()

        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=num_classes)

        self.dp1 = ComplexDropout(0.1)
        self.dp2 = nn.Dropout(0.1)
        self.relu = ComplexReLU()
        self.real_relu = nn.ReLU()


    def forward(self, x):
        x = rearrange(x, "b l (c d) -> b c l d", c = 3)
        x = self.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.maxpool3(x)
        x = rearrange(x, "b c l d -> b ( c l d )")
        x = self.dp1(self.relu(self.fc1(x)))
        x = self.c2r(x)
        x = self.dp2(self.real_relu(self.fc2(x)))
        x = self.fc3(x)
        return x
