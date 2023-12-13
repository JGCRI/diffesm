import os
from typing import Any, Callable

import torch
from accelerate import Accelerator
from diffusers import SchedulerMixin
from omegaconf.dictconfig import DictConfig
from torch.optim import Optimizer
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader
from tqdm import tqdm
from einops import reduce
import wandb
from ema_pytorch import EMA

from utils.viz_utils import create_gif
from data.climate_dataset import ClimateDataset, ClimateDataLoader
from trainers.base_trainer import BaseTrainer
from models.video_net import UNetModel3D


class UNetTrainer(BaseTrainer):
    """Trainer class for 2D diffusion models."""

    def __init__(
        self,
        train_set: ClimateDataset,
        val_set: ClimateDataset,
        model: UNetModel3D,
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

        self.scheduler.set_timesteps(self.sample_steps)

        # Keep track of our exponential moving average weights
        self.ema_model = EMA(
            self.model,
            beta=0.9999,  # exponential moving average factor
            update_after_step=100,  # only after this number of .update() calls will it start updating
            update_every=10,
        ).to(self.accelerator.device)

        # Assign the device and weight dtype (32 bit for training)
        self.device = self.accelerator.device
        self.weight_dtype = torch.float32

        self.optimizer = optimizer(self.model.parameters())

        self.train_loader: ClimateDataLoader = dataloader(
            self.train_set,
            self.accelerator,
            self.batch_size,
        )
        self.val_loader: ClimateDataLoader = dataloader(
            self.val_set,
            self.accelerator,
            self.batch_size,
        )

        # Initialize counters
        self.global_step = 0
        self.first_epoch = 0

        # Keep track of important variables for logging
        self.total_batch_size = (
            self.batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )
        self.num_steps_per_epoch = (
            len(self.train_loader) // self.accelerator.gradient_accumulation_steps
        )
        self.max_train_steps = self.max_epochs * self.num_steps_per_epoch

        # Log to WANDB (on main process only)
        if self.accelerator.is_main_process:
            self.log_hparams()

        # Load model states from checkpoints if they exist
        if self.load_path:
            self.load()

        # Prepare everything for GPU training
        self.prepare()

    def log_hparams(self):
        """Logs the hyperparameters to WANDB."""
        run = self.accelerator.get_tracker("wandb").tracker

        hparam_dict = {
            "Number Training Examples": len(self.train_set),
            "Number Epochs": self.max_epochs,
            "Batch Size per Device": self.batch_size,
            "Total Train Batch Size (w. distributed & accumulation)": self.total_batch_size,
            "Gradient Accumulation Steps": self.accelerator.gradient_accumulation_steps,
            "Total Optimization Steps": self.max_train_steps,
        }

        run.config.update(hparam_dict)

    def prepare(self):
        """Just send all relevant objects through the accelerator to be placed on GPU."""
        (
            self.model,
            self.optimizer,
        ) = self.accelerator.prepare(self.model, self.optimizer)

    def train(self):
        # Sanity check the validation loop and sampling before training
        for epoch in range(self.first_epoch, self.max_epochs):
            
            progress_bar = tqdm(
                total=self.num_steps_per_epoch,
                disable=not self.accelerator.is_local_main_process,
            )
            progress_bar.set_description(f"Epoch {epoch}")

            for step, batch in enumerate(self.train_loader.generate()):
                self.model.train()
                # Skip steps until we reach the resumed step
                if (
                    self.load_path
                    and epoch == self.first_epoch
                    and step < self.resume_step
                ):
                    if step % self.accelerator.gradient_accumulation_steps == 0:
                        progress_bar.update(1)

                loss = self.get_loss(batch)

                # Check if the accelerator has performed an optimization step
                if self.accelerator.sync_gradients:
                    # Update counts
                    progress_bar.update(1)
                    self.global_step += 1
                    self.ema_model.update()

                    if self.accelerator.is_main_process:
                        # Check to see if we need to sample from our model
                        if self.global_step % self.sample_every == 0:
                            self.sample()

                        # Check to see if we need to save our model
                        if self.global_step % self.save_every == 0:
                            self.save()

                # Metric calculation and logging
                log_dict = {"Training/Loss": loss.detach().item()}
                self.accelerator.log(log_dict, step=self.global_step)
                progress_bar.set_postfix(**log_dict)

            progress_bar.close()

    def get_loss(self, batch):
        clean_samples = batch.to(self.weight_dtype)
        cond_map = reduce(clean_samples, "b v t h w -> b v 1 h w", "mean").repeat(
            1, 1, clean_samples.shape[-3], 1, 1
        )

        # Sample noise that we'll add to the clean images
        noise = torch.randn_like(clean_samples)
        timesteps = torch.randint(
            0, len(self.scheduler), (clean_samples.shape[0],), device=self.device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_samples = self.scheduler.add_noise(clean_samples, noise, timesteps)

        with self.accelerator.accumulate(self.model):
            model_output = self.model(
                noisy_samples,
                timesteps,
                cond_map=cond_map,
            )

            # Make sure to get the right target for the loss
            if self.scheduler.config.prediction_type == "epsilon":
                target = noise
            elif self.scheduler.config.prediction_type == "v_prediction":
                target = self.scheduler.get_velocity(clean_samples, noise, timesteps)
            else:
                raise NotImplementedError("Only epsilon and v_prediction supported")

                # Calculate loss and update gradients
            loss = mse_loss(model_output.float(), target.float())
            self.accelerator.backward(loss)

            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss

    @torch.inference_mode()
    def validation_loop(self, sanity_check=False) -> None:
        """Runs a single epoch of validation.

        Updates the loss, logs it, and backpropagates the error.
        """
        self.model.eval()
        val_loss = 0

        for batch_idx, batch in enumerate(self.val_loader.generate()):
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

        self.ema_model.eval()
        # Grab a random sample from validation set
        batch = next(iter(self.val_loader.generate()))[0:1]

        clean_samples = batch.to(self.weight_dtype)
        cond_map = reduce(clean_samples, "b v t h w -> b v 1 h w", "mean").repeat(
            1, 1, clean_samples.shape[-3], 1, 1
        )

        # Sample noise that we'll add to the clean images
        gen_sample = torch.randn_like(clean_samples)

        # Run the diffusion process in reverse
        for i in tqdm(
            range(
                len(self.scheduler) - 1, -1, -len(self.scheduler) // self.sample_steps
            ),
            "Sampling",
        ):
            timestep = torch.tensor([i] * gen_sample.shape[0], device=self.device)
            output = self.ema_model(
                gen_sample,
                timestep,
                cond_map=cond_map,
            )

            gen_sample = self.scheduler.step(
                output, timestep=i, sample=gen_sample
            ).prev_sample

        # Turn the samples into xr datasets
        gen_ds = self.val_set.convert_tensor_to_xarray(gen_sample[0])
        val_ds = self.val_set.convert_tensor_to_xarray(clean_samples[0])

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

    def save(self):
        """Saves the state of training to disk."""
        if self.save_name is None:
            return
        else:
            state_dict = {
                "EMA": self.ema_model,
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
        """Loads the state of training from a checkpoint."""

        # Make sure to map all tensors to the CPU for consistency
        checkpoint = torch.load(
            os.path.join(self.load_dir, self.load_path), map_location="cpu"
        )

        # Load trainer variables
        self.global_step = checkpoint["Global Step"]

        # Update counts related to progress in training
        self.resume_global_step = (
            self.global_step * self.accelerator.gradient_accumulation_steps
        )
        self.first_epoch = self.global_step // self.num_steps_per_epoch
        self.resume_step = self.resume_global_step % (
            self.num_steps_per_epoch * self.accelerator.gradient_accumulation_steps
        )

        # Load the state dict for models and optimizers
        self.model.load_state_dict(checkpoint["Unet"])
        self.optimizer.load_state_dict(checkpoint["Optimizer"])
