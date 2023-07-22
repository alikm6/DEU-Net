import torch
import torch.nn as nn


class ConvBNReLUBlock(nn.Module):
    """convolution => [BN] => ReLU"""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            padding: int = 1
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class DoubleBNReLUBlock(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            mid_channels: int = None,
            conv1_kernel_size: int = 3,
            conv1_padding: int = 1,
            conv2_kernel_size: int = 3,
            conv2_padding: int = 1
    ):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels

        self.conv1 = ConvBNReLUBlock(in_channels, mid_channels, kernel_size=conv1_kernel_size, padding=conv1_padding)
        self.conv2 = ConvBNReLUBlock(mid_channels, out_channels, kernel_size=conv2_kernel_size, padding=conv2_padding)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        return x


class UpSampleBlock(nn.Module):
    """Up Scaling"""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            scale_factor: int = 2,
            bilinear: bool = False
    ):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True),
                ConvBNReLUBlock(in_channels, out_channels, kernel_size=3, padding=1)
            )
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=scale_factor, stride=scale_factor)

    def forward(self, x):
        return self.up(x)


class SEBlock(nn.Module):
    def __init__(self, channel, r=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Fusion
        y = torch.mul(x, y)
        return y


class Permute(nn.Module):
    def __init__(self, *dims):
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


class View(nn.Module):
    def __init__(self, *dims):
        super(View, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.view(*self.dims)
