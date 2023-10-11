from typing import Callable

import hydra
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import MNIST


class TorchvisionDataset(Dataset):
    """Dataset for any torchvision dataset."""

    def __init__(self, data_dir, **kwargs) -> None:
        """Initialize a torchvision dataset.

        :param data_dir: The data directory. Defaults to `"data/"`.
        """
        super().__init__()

        # These are train/val specific keyword arguments
        train_kwargs = kwargs["train_kwargs"]
        self.data = hydra.utils.instantiate(
            kwargs["dataset"], root=data_dir, download=True, **train_kwargs
        )

    @property
    def has_classes(self) -> bool:
        """Return whether the dataset has classes."""
        return getattr(self.data, "classes", None) is not None

    @property
    def num_classes(self) -> int:
        """Return the number of classes."""
        if getattr(self.data, "classes", None) is not None:
            return len(set(self.data.classes))
        else:
            return 0

    @property
    def spatial_size(self) -> tuple[int, int]:
        """Return the spatial size of the images."""
        sample = self.data[0]
        if type(self.data[0] == tuple):
            sample = sample[0]

        return sample.shape[-2:]

    @property
    def channels(self) -> int:
        """Return the number of channels."""
        sample = self.data[0]
        if type(self.data[0] == tuple):
            sample = sample[0]

        return sample.shape[-3]

    def __getitem__(self, index) -> tuple[Tensor, Tensor]:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)


if __name__ == "__main__":
    _ = TorchvisionDataset()
