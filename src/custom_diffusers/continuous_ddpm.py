from functools import partial
from typing import Sequence
import torch.nn.functional as F
from torch.special import expm1
from torch import nn
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMSchedulerOutput

import math
from math import sqrt
from einops import repeat


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


# diffusion helpers


def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))


# continuous schedules

# equations are taken from https://openreview.net/attachment?id=2LdBqxc1Yv&name=supplementary_material
# @crowsonkb Katherine's repository also helped here https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/utils.py

# log(snr) that approximates the original linear schedule


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def beta_linear_log_snr(t):
    return -log(expm1(1e-4 + 10 * (t**2)))


def alpha_cosine_log_snr(t, s=0.008):
    return -log((torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** -2) - 1, eps=1e-5)


def log_snr_to_alpha_sigma(log_snr):
    return torch.sqrt(torch.sigmoid(log_snr)), torch.sqrt(torch.sigmoid(-log_snr))


class Config:
    pass


class ContinuousDDPM(nn.Module):
    def __init__(
        self,
        beta_schedule="linear",
        prediction_type="v_prediction",
    ):
        super().__init__()
        self.prediction_type = prediction_type
        self.config = Config()
        self.config.prediction_type = prediction_type

        if beta_schedule == "linear":
            self.log_snr = beta_linear_log_snr
        elif beta_schedule == "cosine":
            self.log_snr = alpha_cosine_log_snr
        else:
            raise ValueError(f"unknown noise schedule {beta_schedule}")

        # sampling
        self.sample_steps: int = None
        self.timesteps: Sequence[int] = None

    def q_posterior(self, x_start, x_t, t, *, t_next=None):
        t_next = default(t_next, lambda: (t - 1.0 / self.num_timesteps).clamp(min=0.0))

        """ https://openreview.net/attachment?id=2LdBqxc1Yv&name=supplementary_material """
        log_snr = self.log_snr(t)
        log_snr_next = self.log_snr(t_next)
        log_snr, log_snr_next = map(
            partial(right_pad_dims_to, x_t), (log_snr, log_snr_next)
        )

        alpha, sigma = log_snr_to_alpha_sigma(log_snr)
        alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

        # c - as defined near eq 33
        c = -expm1(log_snr - log_snr_next)
        posterior_mean = alpha_next * (x_t * (1 - c) / alpha + c * x_start)

        # following (eq. 33)
        posterior_variance = (sigma_next**2) * c
        posterior_log_variance_clipped = log(posterior_variance, eps=1e-20)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def predict_start_from_noise(self, x_t, t, noise):
        log_snr = self.log_snr(t)
        log_snr = right_pad_dims_to(x_t, log_snr)
        alpha, sigma = log_snr_to_alpha_sigma(log_snr)
        return (x_t - sigma * noise) / alpha.clamp(min=1e-8)

    def predict_start_from_v(self, x_t, t, v):
        log_snr = self.log_snr(t)
        log_snr = right_pad_dims_to(x_t, log_snr)
        alpha, sigma = log_snr_to_alpha_sigma(log_snr)
        return alpha * x_t - sigma * v

    def set_timesteps(self, num_inference_steps):
        self.sample_steps = num_inference_steps
        self.timesteps = torch.arange(self.sample_steps)

    def step(
        self, model_output: torch.FloatTensor, timestep: int, sample: torch.FloatTensor
    ):
        assert (
            self.sample_steps is not None
        ), "You must call set_timesteps before calling step"

        steps = torch.linspace(
            1.0, 0.0, self.sample_steps + 1, device=model_output.device
        )
        times = steps[timestep]
        times_next = steps[timestep + 1]

        # Calculate the SNR for the current timestep
        log_snr = self.log_snr(times)
        log_snr = repeat(log_snr, " -> b", b=sample.shape[0])

        # Predict the x0 from the model output
        x_start = self.predict_start_from_v(sample, times, model_output)
        model_mean, model_variance, _ = self.q_posterior(
            x_start, sample, times, t_next=times_next
        )

        # If we are at the end just return the mean
        if times_next == 0:
            return DDPMSchedulerOutput(model_mean, x_start)

        # Otherwise return the mean with a little bit of noise added
        noise = torch.randn_like(sample)
        prev_sample = model_mean + sqrt(model_variance) * noise
        return DDPMSchedulerOutput(prev_sample, x_start)

    # Length is one to mark this as a continuous scheduler
    def __len__(self):
        return self.sample_steps

    def get_velocity(
        self,
        sample: torch.Tensor,
        noise: torch.FloatTensor,
        timesteps: torch.FloatTensor,
    ):
        """Get the velocity of the sample at the given timesteps"""

        log_snr = timesteps
        log_snr_padded_dim = right_pad_dims_to(sample, log_snr)
        alpha, sigma = log_snr_to_alpha_sigma(log_snr_padded_dim)

        return alpha * noise - sigma * sample

    def add_noise(self, original_samples, noise, timesteps: torch.FloatTensor):
        log_snr = timesteps
        log_snr_padded_dim = right_pad_dims_to(original_samples, log_snr)
        alpha, sigma = log_snr_to_alpha_sigma(log_snr_padded_dim)

        return alpha * original_samples + sigma * noise
