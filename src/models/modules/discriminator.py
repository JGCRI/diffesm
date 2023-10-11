from torch import Tensor, nn


class PatchGANDiscriminator(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim: int,
        num_layers: int = 3,
        kernel_size: int = 4,
        norm_layer="batchnorm",
    ):
        super().__init__()

        match norm_layer:
            case "batchnorm":
                norm = nn.BatchNorm2d
                use_bias = False
            case "instancenorm":
                norm = nn.InstanceNorm2d
                use_bias = True
            case _:
                raise NotImplementedError(f"Norm layer {norm_layer} not implemented")

        self.act = nn.LeakyReLU(0.2)
        self.in_conv = nn.Conv2d(
            dim_in,
            dim,
            kernel_size=kernel_size,
            padding=1,
            stride=2,
        )

        dim_mult = 1
        dim_mult_prev = 1

        self.mid_layers = nn.ModuleList()
        # Gradually increase the number of filters
        for i in range(1, num_layers):
            dim_mult_prev = dim_mult
            dim_mult = min(2**i, 8)

            # Each layer is composed of convolution, batchnorm and then relu
            self.mid_layers.extend(
                [
                    # Don't include bias since batchnorm has affine parameters
                    nn.Conv2d(
                        dim * dim_mult_prev,
                        dim * dim_mult,
                        kernel_size=kernel_size,
                        padding=1,
                        stride=2,
                        bias=use_bias,
                    ),
                    norm(dim * dim_mult),
                    nn.LeakyReLU(0.2),
                ]
            )

        # Construct the last convolutional layer
        dim_mult_prev = dim_mult
        dim_mult = min(2**num_layers, 8)
        self.mid_layers.extend(
            [
                nn.Conv2d(
                    dim * dim_mult_prev,
                    dim * dim_mult,
                    kernel_size=kernel_size,
                    padding=1,
                    stride=1,
                    bias=use_bias,
                ),
                norm(dim * dim_mult),
                nn.LeakyReLU(0.2),
            ]
        )

        # Convert output to 1 channel for prediction
        self.out_conv = nn.Conv2d(dim * dim_mult, 1, kernel_size=kernel_size, padding=1, stride=1)

    def forward(self, x: Tensor):
        x = self.in_conv(x)
        x = self.act(x)

        for layer in self.mid_layers:
            x = layer(x)

        x = self.out_conv(x)

        return x


if __name__ == "__main__":
    # Test the discriminator
    import torch

    model = PatchGANDiscriminator(dim_in=3, dim=64, num_layers=4)

    fake_image = torch.randn(8, 3, 256, 256)

    print(model(fake_image).shape)
