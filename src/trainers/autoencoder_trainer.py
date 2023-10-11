import os
from collections import defaultdict
from typing import Any, Callable

import torch
from accelerate import Accelerator
from omegaconf.dictconfig import DictConfig
from torch import Tensor
from torch._dynamo.eval_frame import OptimizedModule
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

import wandb
from data.celeba_hq_dataset import CelebAHQ
from diffusers import AutoencoderKL
from models.modules.loss import PerceptualAdversarialLoss


class AutoEncoderTrainer:
    """Trainer class for 2D diffusion models."""

    def __init__(
        self,
        train_set: CelebAHQ,
        val_set: CelebAHQ,
        autoencoder: AutoencoderKL,
        loss: PerceptualAdversarialLoss,
        accelerator: Accelerator,
        hyperparameters: DictConfig,
        dataloader: Callable[[Any], DataLoader],
        autoencoder_opt: Callable[[Any], Optimizer],
        disc_opt: Callable[[Any], Optimizer],
    ) -> None:
        # Assign the hyperparameters to class attributes
        self.save_hyperparameters(hyperparameters)

        # Assign more class attributes
        self.accelerator = accelerator
        self.train_set, self.val_set = train_set, val_set
        self.autoencoder = autoencoder

        self.loss = loss

        # We need separate optimizers for the autoencoder and the discriminator
        self.autoencoder_opt = autoencoder_opt(self.autoencoder.parameters())
        self.disc_opt = disc_opt(self.loss.discriminator.parameters())

        self.train_loader = dataloader(self.train_set)
        self.val_loader = dataloader(self.val_set)

        # Initialize counters
        self.global_step = 0
        self.start_epoch = 0

        # Load model states from checkpoints if they exist
        self.load()

        # Prepare everything for GPU training
        self.prepare()

    def prepare(self):
        """Just send all relevant objects through the accelerator to be placed on GPU."""
        (
            self.autoencoder,
            self.loss,
            self.autoencoder_opt,
            self.disc_opt,
            self.train_loader,
            self.val_loader,
        ) = self.accelerator.prepare(
            self.autoencoder,
            self.loss,
            self.autoencoder_opt,
            self.disc_opt,
            self.train_loader,
            self.val_loader,
        )

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

    def get_last_layer(self):
        """Gets the last layer of the autoencoder."""
        if type(self.autoencoder) == DDP or type(self.autoencoder) == OptimizedModule:
            return self.autoencoder.module.decoder.conv_out.weight
        else:
            return self.autoencoder.decoder.conv_out.weight

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

    def training_loop(self, epoch: int) -> None:
        """Runs a single epoch of training.

        Updates the loss, logs it, and backpropagates the error.
        """
        self.autoencoder.train()
        self.loss.train()

        # Only display progress bar on main process
        pbar = tqdm(
            self.train_loader,
            unit="batch",
            desc=f"Training Epoch {epoch}",
            disable=not self.accelerator.is_main_process,
        )
        for batch_idx, batch in enumerate(pbar):
            if epoch == self.start_epoch and batch_idx < (
                self.global_step % len(self.train_loader)
            ):
                continue
            batch = self.normalize(batch)
            reconstructions, posterior = self.model_forward_pass(batch)

            ae_loss, disc_loss, loss_dict = self.loss(
                batch,
                reconstructions,
                posterior,
                self.global_step,
                last_layer=self.get_last_layer(),
            )

            # Update the loss metric and take an update step
            self.global_step += 1
            self.accelerator.log(
                {"Training/Total Loss": ae_loss, "Training/Disc Loss": disc_loss, "Epoch": epoch}
                | {f"Training/{metric}": value.item() for metric, value in loss_dict.items()},
                step=self.global_step,
            )

            # Gradient Update
            self.autoencoder_opt.zero_grad()
            self.accelerator.backward(ae_loss)
            self.autoencoder_opt.step()

            self.disc_opt.zero_grad()
            self.accelerator.backward(disc_loss)
            self.disc_opt.step()

            pbar.set_postfix(
                {
                    "Training AE Loss": ae_loss.item(),
                }
            )

    @torch.inference_mode()
    def validation_loop(self, epoch: int, sanity_check=False) -> None:
        """Runs a single epoch of validation.

        Updates the loss, logs it, and backpropagates the error.
        """
        self.autoencoder.eval()
        self.loss.eval()

        # Keep track of running losses
        val_metric_dict = defaultdict(lambda: 0)

        # Only display progress bar on main process
        pbar = tqdm(
            self.val_loader,
            unit="batch",
            desc=f"Validation Epoch {epoch}" if not sanity_check else "Sanity Check",
            disable=not self.accelerator.is_main_process,
        )
        # Iterate over the batches
        for batch_idx, batch in enumerate(pbar):
            # If we are sanity checking, only run 10 batches
            if sanity_check and batch_idx > 10:
                return

            batch = self.normalize(batch)
            reconstructions, posterior = self.model_forward_pass(batch)

            _, _, loss_dict = self.loss(
                batch,
                reconstructions,
                posterior,
                global_step=self.global_step,
                last_layer=self.get_last_layer(),
            )

            # Update the loss metric
            for metric, value in loss_dict.items():
                # Gather the metric across all processes and take the average
                val_metric_dict["Validation/" + metric] += self.accelerator.reduce(
                    value, reduction="mean"
                ).item()

            pbar.set_postfix({"Validation Loss": loss_dict["Total Loss"]})

        # Take the average of all of the metrics and log them
        for metric, value in val_metric_dict.items():
            val_metric_dict[metric] = value / len(self.val_loader)
        self.accelerator.log(val_metric_dict, step=self.global_step)

    @torch.inference_mode()
    def sample(self) -> None:
        """Samples a batch of images from the model."""

        # Get 5 random images from the validation set
        batch = next(iter(self.val_loader))
        batch = batch[:5]

        reconstruction, posterior = self.model_forward_pass(self.normalize(batch))

        latent_mean = self.denormalize(posterior.mode())

        pred_x = self.denormalize(reconstruction).clamp(min=0, max=1)

        stacked_images = torch.cat([batch, pred_x], dim=0)

        grid = make_grid(stacked_images, nrow=5)
        self.accelerator.log({"Reconstructions": wandb.Image(grid)}, step=self.global_step)
        self.accelerator.log({"Latent Space": wandb.Image(latent_mean)}, step=self.global_step)

    def model_forward_pass(self, batch: tuple[Tensor, Tensor] | Tensor) -> tuple[Tensor, Tensor]:
        """Given a single batch, sends it through the diffuser to obtain a loss and then
        backpropagates the loss."""

        if type(self.autoencoder) == DDP or type(self.autoencoder) == OptimizedModule:
            posterior = self.autoencoder.module.tiled_encode(batch).latent_dist
            reconstructions = self.autoencoder.module.tiled_decode(posterior.mean).sample
        else:
            posterior = self.autoencoder.tiled_encode(batch).latent_dist
            reconstructions = self.autoencoder.tiled_decode(posterior.mean).sample

        # Only care about the float now
        return reconstructions, posterior

    def save(self, epoch: int):
        """Saves the state of training to disk."""
        if self.save_name is None:
            return
        else:
            # Wait for all processes to reach this point
            self.accelerator.wait_for_everyone()
            state_dict = {
                "Autoencoder": self.accelerator.unwrap_model(self.autoencoder).state_dict(),
                "Discriminator": self.accelerator.unwrap_model(
                    self.loss
                ).discriminator.state_dict(),
                "Autoencoder Optimizer": self.autoencoder_opt.state_dict(),
                "Discriminator Optimizer": self.disc_opt.state_dict(),
                "Epoch": epoch + 1,
                "Global Step": self.global_step,
            }

            # If the directory doesn't exist already create it
            os.makedirs(self.save_dir, exist_ok=True)

            # Add an extension if one doesn't exist
            save_name = self.save_name if "." in self.save_name else self.save_name + ".pt"

            # Every 10 epochs, append the epoch number to save it
            if epoch % 10 == 0:
                save_split = save_name.split(".")
                save_split[0] += "_" + str(epoch)
                save_name = ".".join(save_split)

            # Save the State dictionary to disk
            self.accelerator.save(state_dict, os.path.join(self.save_dir, save_name))

    def load(self):
        """Loads the state of trainin from a checkpoint."""
        if self.load_name is None:
            return
        else:
            # Add an extension if one doesn't exist
            load_name = self.load_name if "." in self.load_name else self.load_name + ".pt"
            state_dict = torch.load(os.path.join(self.load_dir, load_name), map_location="cpu")

            # Load trainer variables
            self.start_epoch = state_dict["Epoch"]
            self.global_step = state_dict["Global Step"]

            # Load the state dict for models and optimizers
            self.loss.discriminator.load_state_dict(state_dict["Discriminator"])
            self.autoencoder.load_state_dict(state_dict["Autoencoder"])
            self.autoencoder_opt.load_state_dict(state_dict["Autoencoder Optimizer"])
            self.disc_opt.load_state_dict(state_dict["Discriminator Optimizer"])
