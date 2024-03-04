from collections import namedtuple

from torch import nn

ResNetParams = namedtuple(
    "ResNetParams", "in_channels,out_channels,kernel_sizes,strides"
)


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        """
        If in_channels == out_channels, just add identity map of input;
        otherwise apply conv kernel of size 1 without padding to get
        the appropriate shape to be added in the skip connection.
        Works for any stride, since padding in main conv is kernel_size // 2,
        which keeps the numerator in the conv dim formula constant for dilation=1.

        Conv dim formula:
            new_dim = floor(
                (
                    old_dim + 2 * p - (k - 1) * (d - 1) - k
                ) / stride
            ) + 1
        where p is padding on each side, k is kernel_size and d is dilation (always
        assumed to be 1).

        Usually, makes sense for stride=1 if in_channels == out_channels
        and stride=2 if out_channels = 2 * in_channels.
        """
        # kernel_size has to be odd to have easy formula and dim reduction;
        assert kernel_size % 2
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        # for padding 1 and kernel size 3 the conv dim formula gives numerator
        # in_dim + 2 * pad - ker_dim = in_dim - 1
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
            ),
            nn.BatchNorm2d(
                out_channels, track_running_stats=False, affine=True
            ),
            nn.ReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
        )
        # here padding is 0 and kernel size is 1 so in numerator of
        # conv dim formula you get in_dim + full_pad - kernel = in_dim - 1
        if in_channels != out_channels:
            self.reshape_input = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride
            )

    def forward(self, x):
        res = self.net(x)
        if self.in_channels != self.out_channels:
            return self.reshape_input(x) + res
        return res + x


class ResNet(nn.Module):
    def __init__(self, res_block_config):
        """
        Sequential application of ResNetBlock modules.

        Output is from conv layer, not normalised and no activation applied.
        """
        super().__init__()
        self.net = nn.Sequential()
        for i in range(len(res_block_config.in_channels)):
            self.net.add_module(
                f"ResBlock_{i}",
                ResNetBlock(
                    in_channels=res_block_config.in_channels[i],
                    out_channels=res_block_config.out_channels[i],
                    kernel_size=res_block_config.kernel_sizes[i],
                    stride=res_block_config.strides[i],
                ),
            )
            if i < len(res_block_config.in_channels) - 1:
                self.net.add_module(
                    f"BN2d_{i}",
                    nn.BatchNorm2d(
                        num_features=res_block_config.out_channels[i],
                        affine=True,
                        track_running_stats=False,
                    ),
                )
                self.net.add_module(f"ReLU_{i}", nn.ReLU())

    def forward(self, x):
        return self.net(x)
