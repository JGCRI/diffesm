from abc import ABC, abstractmethod

from omegaconf import DictConfig
from torch import Tensor


class BaseTrainer(ABC):
    @abstractmethod
    def prepare(self) -> None:
        pass

    def save_hyperparameters(self, cfg: DictConfig) -> None:
        """Saves the hyperparameters as class attributes."""
        for key, value in cfg.items():
            setattr(self, key, value)

    def normalize(self, batch: Tensor) -> Tensor:
        """Normalizes the batch to be between -1 and 1."""
        return batch * 2 - 1

    def denormalize(self, batch: Tensor) -> Tensor:
        """Denormalizes the batch to be between 0 and 1."""
        return (batch + 1) / 2

    def train(self) -> None:
        """Runs the main training loop."""

        # Sanity check the validation loop and sampling before training
        self.validation_loop(epoch=0, sanity_check=True)
        self.sample()

        for epoch in range(self.start_epoch, self.epochs):
            if not self.skip_training:
                self.training_loop(epoch)
            self.validation_loop(epoch)
            self.sample()
            self.save(epoch)

    @abstractmethod
    def training_loop(self, epoch: int) -> None:
        pass

    @abstractmethod
    def validation_loop(self, epoch: int, sanity_check: bool = False) -> None:
        pass

    @abstractmethod
    def sample(self) -> None:
        pass

    @abstractmethod
    def model_forward_pass(self, batch: Tensor) -> Tensor:
        pass

    @abstractmethod
    def save(self, epoch: int) -> None:
        pass

    @abstractmethod
    def load(self) -> None:
        pass
