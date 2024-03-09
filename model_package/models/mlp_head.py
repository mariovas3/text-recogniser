import torch
from torch import nn


class MLPHead(nn.Module):
    def __init__(self, in_channels, flattened_dim, followup_dim, out_dim):
        """
        Assumes input is Conv2d output of some convnet (no BN or activations applied).
        """
        super().__init__()

        self.conv_lvl_layers = nn.Sequential(
            nn.BatchNorm2d(
                in_channels, affine=True, track_running_stats=False
            ),
            nn.ReLU(),
        )
        self.net = nn.Sequential(
            nn.Linear(flattened_dim, followup_dim),
            nn.BatchNorm1d(
                followup_dim, affine=True, track_running_stats=False
            ),
            nn.ReLU(),
            nn.Linear(followup_dim, out_dim),
        )

    def forward(self, x):
        x = self.conv_lvl_layers(x)
        return self.net(torch.flatten(x, 1))
