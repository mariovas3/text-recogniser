from torch import nn

# this config is default to pair up with transformer decoder
# works well to get cer=0 on single 64-sized batch after
# 200 steps and lr=1e-3.
RESNET_CONFIG = dict(
    in_channels=[1, 4, 4, 8, 8, 16, 16, 32, 32],
    out_channels=[4, 4, 8, 8, 16, 16, 32, 32, 64],
    kernel_sizes=[5, 3, 3, 3, 3, 3, 3, 3, 3],
    strides=[2, 1, 2, 1, 2, 1, 2, 1, 2],
)


def get_final_height_width(h_in, w_in, resnet_config: dict):
    K = len(resnet_config["kernel_sizes"])
    paddings = (1,) * K
    h_out = conv_dim_formula(
        h_in,
        resnet_config["kernel_sizes"],
        paddings,
        resnet_config["strides"],
    )
    w_out = conv_dim_formula(
        w_in,
        resnet_config["kernel_sizes"],
        paddings,
        resnet_config["strides"],
    )
    return h_out, w_out


def conv_dim_formula(in_dim, kernels, paddings, strides, dilations=None):
    """
    Calculate final shape of in_dim after successive conv1d operations.

    Args:
        in_dim (int): input dimension.
        kernels (Sequence[int]): indexable seq of kernels.
        paddings (Sequence[int]): indexable seq of padding on both sides.
        strides (Sequence[int]): indexable seq of strides.
        dilations (Sequence[int]): indexable seq of dilations.
            If dilations is None, all dilations assumed to be 1.
    """
    assert len(kernels) == len(paddings) == len(strides)
    if dilations is not None:
        assert len(kernels) == len(dilations)

    out = in_dim
    for i in range(len(kernels)):
        offset = 2 * paddings[i] - kernels[i]
        if dilations is not None:
            offset -= (kernels[i] - 1) * (dilations[i] - 1)
        out = (out + offset) // strides[i] + 1
    return out


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
    def __init__(self, resnet_config):
        """
        Sequential application of ResNetBlock modules.

        Output is from conv layer, not normalised and no activation applied.
        """
        super().__init__()
        self.net = nn.Sequential()
        for i in range(len(resnet_config["in_channels"])):
            self.net.add_module(
                f"ResBlock_{i}",
                ResNetBlock(
                    in_channels=resnet_config["in_channels"][i],
                    out_channels=resnet_config["out_channels"][i],
                    kernel_size=resnet_config["kernel_sizes"][i],
                    stride=resnet_config["strides"][i],
                ),
            )
            if i < len(resnet_config["in_channels"]) - 1:
                self.net.add_module(
                    f"BN2d_{i}",
                    nn.BatchNorm2d(
                        num_features=resnet_config["out_channels"][i],
                        affine=True,
                        track_running_stats=False,
                    ),
                )
                self.net.add_module(f"ReLU_{i}", nn.ReLU())

    def forward(self, x):
        return self.net(x)


class ResNetClassifier(nn.Module):
    def __init__(self, resnet, mlp):
        """
        Combine ResNet instance with MLP classification head.
        """
        super().__init__()
        self.resnet, self.mlp = resnet, mlp

    def forward(self, x):
        return self.mlp(self.resnet(x))
