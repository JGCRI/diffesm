import torch
from torch import nn


class MinUNet(nn.Module):
    """A minimal U-Net architecture for image denoising."""

    def __init__(self) -> None:
        super().__init__()

        # Helper function to create a convolutional block
        def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(),
            )

        self.blocks = nn.Sequential(
            conv_block(1, 64),
            conv_block(64, 128),
            conv_block(128, 256),
            conv_block(256, 512),
            conv_block(512, 256),
            conv_block(256, 128),
            conv_block(128, 64),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass x through the blocks."""
        return self.blocks(x)
