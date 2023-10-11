import torch
from accelerate import Accelerator
from einops import rearrange
from hydra.utils import instantiate
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from data.torchvision_dataset import TorchvisionDataset
from src.diffusers.gaussian import GaussianDiffusion
from src.utils.logger import MetricLogger

Dataset = TorchvisionDataset
Diffuser = GaussianDiffusion


class Trainer2D:
    """Trainer class for 2D diffusion models."""

    def __init__(
        self,
        train_set: Dataset,
        val_set: Dataset,
        diffuser: Diffuser,
        accelerator: Accelerator,
        cfg: DictConfig,
        **_,
    ) -> None:
        # Setup accelerator and logging
        self.accelerator = accelerator
        self.setup(cfg)

        self.cfg = cfg.trainer
        self.train_set, self.val_set = train_set, val_set
        self.model = diffuser
        self.optimizer = instantiate(self.cfg.optimizer, params=self.model.parameters())

        self.train_loader, self.val_loader = self.create_dataloaders()

        # Prepare everything for CUDA
        (
            self.model,
            self.optimizer,
            self.train_loader,
            self.val_loader,
        ) = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.val_loader
        )

    def create_dataloaders(self):
        """Define how the dataloaders should behave."""
        train_loader = DataLoader(self.train_set, **self.cfg.dataloader)
        val_loader = DataLoader(self.val_set, **self.cfg.dataloader)

        return train_loader, val_loader

    def setup(self, cfg):
        """Performs any necessary setup before training begins."""
        log_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        self.accelerator.init_trackers(
            cfg.wandb.project_name,
            log_config,
            init_kwargs={"wandb": cfg.wandb.init_kwargs},
        )

        # Construct the metric logger and add the relevant metrics
        self.metric_logger = MetricLogger(self.accelerator, cfg.trainer.log_interval)
        self.metric_logger.add_metric("Training Loss", on_step=True, on_epoch=True)
        self.metric_logger.add_metric("Validation Loss", on_step=False, on_epoch=True)

    def train(self) -> None:
        """Runs the main training loop."""

        # Make sure validation loop works before training
        self.validation_loop(epoch=0, sanity_check=True)

        for epoch in range(self.cfg.epochs):
            if not self.cfg.skip_training:
                self.training_loop(epoch)
                self.validation_loop(epoch)
            self.sample()

    def training_loop(self, epoch: int) -> None:
        """Runs a single epoch of training.

        Updates the loss, logs it, and backpropagates the error.
        """
        self.model.train()

        with tqdm(
            self.train_loader,
            unit="batch",
            disable=not self.accelerator.is_main_process,
        ) as tbar:
            for batch in tbar:
                tbar.set_description(f"Training Epoch {epoch}")
                loss = self.model_forward_pass(batch)

                # Update the loss metric and take an update step
                self.metric_logger.step()
                self.metric_logger.log("Training Loss", loss.item())

                # Gradient Update
                self.optimizer.zero_grad()
                self.accelerator.backward(loss)
                self.optimizer.step()

                tbar.set_postfix(
                    {"Training Loss": f"{self.metric_logger.compute('Training Loss'):.3f}"}
                )

        # Increment the epoch counter and log the epoch loss
        self.metric_logger.epoch_end("Training Loss")
        self.metric_logger.epoch += 1

    @torch.inference_mode()
    def validation_loop(self, epoch: int, sanity_check=False) -> None:
        """Runs a single epoch of validation.

        Updates the loss, logs it, and backpropagates the error.
        """
        self.model.eval()

        # Run a progress bar over the epoch
        with tqdm(
            self.val_loader,
            unit="batch",
            disable=not self.accelerator.is_main_process,
            total=None if not sanity_check else 10,
        ) as tbar:
            # Iterate over the batches
            for batch_idx, batch in enumerate(tbar):
                # If we are sanity checking, only run 10 batches
                if sanity_check and batch_idx > 10:
                    self.metric_logger.reset("Validation Loss")
                    return

                tbar.set_description(
                    f"Validating Epoch {epoch}" if not sanity_check else "Validation Sanity Check"
                )
                loss = self.model_forward_pass(batch)

                self.metric_logger.log("Validation Loss", loss.item())

                tbar.set_postfix(
                    {"Validation Loss": f"{self.metric_logger.compute('Validation Loss'):.3f}"}
                )

        self.metric_logger.epoch_end("Validation Loss")

    @torch.inference_mode()
    def sample(self) -> None:
        """Samples a batch of images from the model."""

        samples, x0s = self.model.sample(9, return_all_timesteps=True)

        # Select 10 evenly spaced timesteps
        samples = samples[:, torch.linspace(0, samples.shape[1] - 1, 10).long()]
        x0s = x0s[:, torch.linspace(0, x0s.shape[1] - 1, 10).long()]

        samples = rearrange(samples, "b t c h w -> (b h) (t w) c").cpu().numpy()
        x0s = rearrange(x0s, "b t c h w -> (b h) (t w) c").cpu().numpy()

        self.accelerator.log(
            {"Generated Samples": wandb.Image(samples.clip(min=0, max=1), mode="L")}
        )
        self.accelerator.log({"Predicted X_0s": wandb.Image(x0s, mode="L")})

    def model_forward_pass(self, batch: tuple[Tensor, Tensor] | Tensor) -> float:
        """Given a single batch, sends it through the diffuser to obtain a loss and then
        backpropagates the loss."""

        batch, _ = batch

        # Send the batch through the diffuser
        loss = self.model(batch)

        # Only care about the float now
        return loss
