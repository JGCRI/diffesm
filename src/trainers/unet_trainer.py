import os
from typing import Any, Callable, Optional

import torch
from accelerate import Accelerator
from diffusers import SchedulerMixin, DDPMPipeline
from imagen_pytorch import Unet3D
from omegaconf.dictconfig import DictConfig
from torch import Tensor
from torch.optim import Optimizer
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader
from tqdm import tqdm
from einops import reduce
import wandb

from utils.viz_utils import create_gif, plot_map
from data.climate_dataset import ClimateDataset, ClimateDataLoader
from trainers.base_trainer import BaseTrainer


class UNetTrainer(BaseTrainer):
    """Trainer class for 2D diffusion models."""

    def __init__(
        self,
        train_set: ClimateDataset,
        val_set: ClimateDataset,
        model: Unet3D,
        scheduler: SchedulerMixin,
        accelerator: Accelerator,
        hyperparameters: DictConfig,
        dataloader: Callable[[Any], DataLoader],
        optimizer: Callable[[Any], Optimizer],
    ) -> None:
        # Assign the hyperparameters to class attributes
        self.save_hyperparameters(hyperparameters)

        # Assign more class attributes
        self.accelerator = accelerator
        self.train_set, self.val_set = train_set, val_set
        self.model = model
        self.scheduler: SchedulerMixin = scheduler

        # We need separate optimizers for the autoencoder and the discriminator
        self.optimizer = optimizer(self.model.parameters())

        self.train_loader: ClimateDataLoader = dataloader(
            self.train_set, self.accelerator
        )
        self.val_loader: ClimateDataLoader = dataloader(self.val_set, self.accelerator)

        # Initialize counters
        self.global_step = 0

        # Load model states from checkpoints if they exist
        self.load()

        # Prepare everything for GPU training
        self.prepare()

    def prepare(self):
        """Just send all relevant objects through the accelerator to be placed on GPU."""
        (
            self.model,
            self.optimizer,
        ) = self.accelerator.prepare(self.model, self.optimizer)

    def train(self):
        # Sanity check the validation loop and sampling before training
        self.validation_loop(sanity_check=True)
        self.sample()

        while self.global_step < self.max_steps:
            # Main Training Loop
            for batch_idx, batch in enumerate(
                self.train_loader.generate(desc="Training")
            ):
                self.model.train()
                # Send the batch through our autoencoder to obtain loss
                loss, _ = self.model_forward_pass(batch)

                # Update the loss metric and take an update step
                self.global_step += 1
                self.accelerator.log(
                    {
                        "Training/Loss": loss.item(),
                        "Epoch": self.global_step // len(self.train_loader),
                    },
                    step=self.global_step,
                )

                # Gradient Update
                self.optimizer.zero_grad()
                self.accelerator.backward(loss)
                self.optimizer.step()

                # Check to see if we need to validate and sample
                if self.global_step % self.val_every == 0:
                    # self.validation_loop()
                    self.sample()
                    self.save()

    @torch.inference_mode()
    def validation_loop(self, sanity_check=False) -> None:
        """Runs a single epoch of validation.

        Updates the loss, logs it, and backpropagates the error.
        """
        self.model.eval()
        val_loss = 0

        for batch_idx, batch in enumerate(self.val_loader.generate(desc=f"Validating")):
            # If we are sanity checking, only run 10 batches
            if sanity_check and batch_idx > 10:
                return

            val_loss += self.model_forward_pass(batch)[0].item()

        # Log the average
        self.accelerator.log(
            {"Validation/Loss": val_loss / len(self.val_loader)}, step=self.global_step
        )

    @torch.inference_mode()
    def sample(self) -> None:
        """Samples a batch of images from the model."""

        # Grab a random sample from validation set
        val_sample = next(
            iter(self.val_loader.generate(desc="Sampling", disable=True))
        )[0].unsqueeze(0)

        cond_map = reduce(val_sample, "b v t h w -> b v h w", "mean")

        gen_sample = torch.randn_like(val_sample)

        for i in tqdm(range(len(self.scheduler) - 1, -1, -1), "Sampling"):
            timestep = torch.tensor([i] * val_sample.shape[0], device=val_sample.device)
            output = self.model(
                gen_sample,
                timestep,
                cond_images=cond_map,
                ignore_time=True if self.global_step < self.warm_up else False,
            )

            gen_sample = self.scheduler.step(
                output, timestep=i, sample=gen_sample
            ).prev_sample

        # Turn the samples into xr datasets
        gen_ds = self.val_set.convert_tensor_to_xarray(gen_sample[0])
        val_ds = self.val_set.convert_tensor_to_xarray(val_sample[0])

        # Create a gif of the samples
        gen_frames = create_gif(gen_ds)
        val_frames = create_gif(val_ds)

        # Log the gif to wandb
        for var, gif in gen_frames.items():
            self.accelerator.log(
                {f"Generated {var}": wandb.Video(gif, fps=4)}, step=self.global_step
            )

        for var, gif in val_frames.items():
            self.accelerator.log(
                {f"Original {var}": wandb.Video(gif, fps=4)}, step=self.global_step
            )

    def model_forward_pass(
        self, batch: Tensor, timesteps: Optional[int] = None
    ) -> tuple[Tensor, Tensor]:
        # Batch is of shape (batch_size, n_vars, seq_len, n_lat, n_lon)
        # Average the batch along time and repeat it to match the shape of the batch
        cond_map = reduce(batch, "b v t h w -> b v h w", "mean")

        # Noise the input batch
        noise = torch.randn_like(batch)
        timesteps = (
            torch.randint(
                0, len(self.scheduler), (batch.shape[0],), device=batch.device
            )
            if timesteps is None
            else timesteps
        )

        noisy_samples = self.scheduler.add_noise(
            batch, noise=noise, timesteps=timesteps
        )

        # Concatenate the noise and the condition map
        # noisy_samples = torch.cat([noisy_samples, cond_map], dim=1)

        # Pass the noisy samples through the model
        model_output = self.model(
            noisy_samples,
            timesteps,
            cond_images=cond_map,
            ignore_time=True if self.global_step < self.warm_up else False,
        )

        loss = mse_loss(model_output, noise)

        return loss, model_output

    def save(self):
        """Saves the state of training to disk."""
        if self.save_name is None:
            return
        else:
            # Wait for all processes to reach this point
            self.accelerator.wait_for_everyone()
            state_dict = {
                "Unet": self.accelerator.unwrap_model(self.model).state_dict(),
                "Optimizer": self.optimizer.state_dict(),
                "Global Step": self.global_step,
            }

            # If the directory doesn't exist already create it
            os.makedirs(self.save_dir, exist_ok=True)

            # Add an extension if one doesn't exist
            save_name = (
                self.save_name if "." in self.save_name else self.save_name + ".pt"
            )

            # Save the State dictionary to disk
            self.accelerator.save(state_dict, os.path.join(self.save_dir, save_name))

    def load(self):
        """Loads the state of trainin from a checkpoint."""
        if self.load_name is None:
            return
        else:
            # Add an extension if one doesn't exist
            load_name = (
                self.load_name if "." in self.load_name else self.load_name + ".pt"
            )
            state_dict = torch.load(
                os.path.join(self.load_dir, load_name), map_location="cpu"
            )

            # Load trainer variables
            self.global_step = state_dict["Global Step"]

            # Load the state dict for models and optimizers
            self.model.load_state_dict(state_dict["Unet"])
            self.optimizer.load_state_dict(state_dict["Optimizer"])
