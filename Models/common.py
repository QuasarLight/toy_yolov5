import torch
import torch.nn as nn

class CBS(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=0):  # ch_in, ch_out, kernel, stride, padding
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.silu = nn.SiLU()

    def forward(self, x):
        return self.silu(self.bn(self.conv(x)))

class ResUnit(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True,e=0.5):  # ch_in, ch_out, shortcut, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cbs1 = CBS(c1, c_, 1, 1, 0)
        self.cbs2 = CBS(c_, c2, 3, 1, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cbs2(self.cbs1(x)) if self.add else self.cbs2(self.cbs1(x))

class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5):  # ch_in, ch_out, number, shortcut, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cbs1 = CBS(c1, c_, 1, 1, 0)
        self.cbs2 = CBS(c1, c_, 1, 1, 0)
        self.cbs3 = CBS(2 * c_, c2, 1, 1, 0)
        self.resunit_n = nn.Sequential(*(ResUnit(c_, c_, shortcut, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cbs3(torch.cat((self.resunit_n(self.cbs1(x)), self.cbs2(x)), dim=1))

class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cbs1 = CBS(c1, c_, 1, 1, 0)
        self.cbs2 = CBS(c_ * 4, c2, 1, 1, 0)
        self.maxpool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cbs1(x)
        y1 = self.maxpool(x)
        y2 = self.maxpool(y1)
        return self.cbs2(torch.cat([x, y1, y2, self.maxpool(y2)], 1))

class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)