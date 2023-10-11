import math
from functools import partial
from random import random
from typing import Any, Optional

import torch
from einops import rearrange, reduce
from torch import Tensor, nn
from torch.cuda.amp import autocast
from torch.nn import functional as F
from tqdm import tqdm

from src.utils.diffusion_utils import (
    ModelPrediction,
    default,
    extract,
    identity,
    normalize_to_neg_one_to_one,
    unnormalize_to_zero_to_one,
)


def linear_beta_schedule(
    timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02
) -> torch.Tensor:
    """Linear schedule, proposed in original ddpm paper."""
    scale = 1000 / timesteps
    beta_start = scale * beta_start
    beta_end = scale * beta_end
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(
    timesteps: int, start: int = -3, end: int = 3, tau: int = 1
) -> torch.Tensor:
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion(nn.Module):
    """Gaussian Diffusion Model."""

    def __init__(
        self,
        model: nn.Module,
        image_size: tuple[int, int],
        channels: int,
        timesteps: int = 1000,
        sampling_timesteps: Optional[int] = None,
        objective: str = "pred_v",
        beta_schedule: str = "sigmoid",
        schedule_fn_kwargs: dict[str, Any] = dict(),
        ddim_sampling_eta: float = 0.0,
        auto_normalize: bool = True,
        offset_noise_strength: float = 0.0,  # https://www.crosslabs.org/blog/diffusion-with-offset-noise
        min_snr_loss_weight: bool = False,  # https://arxiv.org/abs/2303.09556
        min_snr_gamma: int = 5,
    ) -> None:
        super().__init__()

        self.model = model

        self.channels = channels
        self.self_condition = False  # CHANGE THIS STAT

        self.image_size = image_size

        self.objective = objective

        assert objective in {
            "pred_noise",
            "pred_x0",
            "pred_v",
        }, "objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])"

        if beta_schedule == "linear":
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == "cosine":
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == "sigmoid":
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f"unknown beta schedule {beta_schedule}")

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = default(
            sampling_timesteps, timesteps
        )  # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32
        def register_buffer(name: str, val: Any) -> None:
            return self.register_buffer(name, val.to(torch.float32))

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer("posterior_variance", posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # offset noise strength - in blogpost, they claimed 0.1 was ideal

        self.offset_noise_strength = offset_noise_strength

        # derive loss weight
        # snr - signal noise ratio

        snr = alphas_cumprod / (1 - alphas_cumprod)

        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max=min_snr_gamma)

        if objective == "pred_noise":
            register_buffer("loss_weight", maybe_clipped_snr / snr)
        elif objective == "pred_x0":
            register_buffer("loss_weight", maybe_clipped_snr)
        elif objective == "pred_v":
            register_buffer("loss_weight", maybe_clipped_snr / (snr + 1))

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    @property
    def device(self) -> torch.device:
        """Returns the device."""
        return self.betas.device

    def predict_start_from_noise(self, x_t: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        """Predicts x_0 given x_t and noise."""
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t: Tensor, t: Tensor, x0: Tensor) -> Tensor:
        """Predicts the noise given x_t and x_0."""
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / extract(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
        )

    def predict_v(self, x_start: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        """Predicts v given x_0 and noise."""
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t: Tensor, t: Tensor, v: Tensor) -> Tensor:
        """Predicts x_0 from v, x_t."""
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start: Tensor, x_t: Tensor, t: Tensor) -> Tensor:
        """Calculates the posterior q(x_{t-1} | x_t, x_0)

        Args:
            x_start (Tensor): The starting input
            x_t (Tensor): The noisy input
            t (Tensor): The timestep

        Returns:
            Tensor: x_{t-1}
        """
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(
        self,
        x: Tensor,
        t: Tensor,
        x_self_cond: Optional[Tensor] = None,
        clip_x_start: bool = False,
        rederive_pred_noise: bool = False,
    ) -> ModelPrediction:
        """Given a noisy image, and a timestep, predicts the noise and x_0.

        Args:
            x (Tensor): Noisy input
            t (Tensor): Timestep
            x_self_cond (Optional[Tensor]): Self conditioning. Defaults to None.
            clip_x_start (bool, optional): Whether to clip the prediction from -1 to 1. Defaults to False.
            rederive_pred_noise (bool, optional): Whether to rederive noise. Defaults to False.

        Returns:
            ModelPrediction: x_0 and noise of t-1
        """
        model_output = self.model(x, t, x_self_cond)
        maybe_clip = partial(torch.clamp, min=-1.0, max=1.0) if clip_x_start else identity

        if self.objective == "pred_noise":
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == "pred_x0":
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == "pred_v":
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(
        self,
        x: Tensor,
        t: Tensor,
        x_self_cond: Optional[Tensor] = None,
        clip_denoised: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Calculates the mean and variance of the posterior.

        Args:
            x (Tensor): x_t
            t (Tensor): tensor of timesteps
            x_self_cond (Optional[Tensor], optional): self conditioning. Defaults to None.
            clip_denoised (bool, optional): Whether to clip predictions. Defaults to True.

        Returns:
            tuple[Tensor, Tensor, Tensor, Tensor]: mean, variance, log_variance and x_0
        """
        preds = self.model_predictions(x, t, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_start, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.inference_mode()
    def p_sample(
        self, x: Tensor, t: int, x_self_cond: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
        """Samples x_{t-1} | x_t.

        Args:
            x (Tensor): x_t
            t (int): timestep
            x_self_cond (Optional[Tensor], optional): self conditioning. Defaults to None.

        Returns:
           tuple[Tensor, Tensor]: The predicted image and x_0
        """
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device=device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x, t=batched_times, x_self_cond=x_self_cond, clip_denoised=True
        )
        noise = torch.randn_like(x) if t > 0 else 0.0  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.inference_mode()
    def p_sample_loop(self, shape: list[int], return_all_timesteps: bool = False) -> Tensor:
        """Iteratively denoises from x_0 to x_T.

        Args:
            shape (list[int]): What shape to generate
            return_all_timesteps (bool, optional): Whether to return all noising timesteps. Defaults to False.

        Returns:
            Tensor: Either the single image or all timesteps
        """
        batch, device = shape[0], self.device

        img = torch.randn(shape, device=device)
        imgs = [img]
        x_start_list = []

        x_start = None

        for t in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="sampling loop time step",
            total=self.num_timesteps,
        ):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond)
            x_start_list.append(x_start)
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim=1)
        x_start_list = torch.stack(x_start_list, dim=1)

        ret = self.unnormalize(ret)
        x_start_list = self.unnormalize(x_start_list)
        return ret, x_start_list

    @torch.inference_mode()
    def ddim_sample(self, shape: list[int], return_all_timesteps: bool = False) -> Tensor:
        batch, device, total_timesteps, sampling_timesteps, eta, objective = (
            shape[0],
            self.device,
            self.num_timesteps,
            self.sampling_timesteps,
            self.ddim_sampling_eta,
            self.objective,
        )

        times = torch.linspace(
            -1, total_timesteps - 1, steps=sampling_timesteps + 1
        )  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(
            zip(times[:-1], times[1:])
        )  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device=device)
        imgs = [img]
        x_start_list = []

        x_start = None

        for time, time_next in tqdm(time_pairs, desc="sampling loop time step"):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(
                img, time_cond, self_cond, clip_x_start=True, rederive_pred_noise=True
            )

            if time_next < 0:
                img = x_start
                imgs.append(img)
                x_start_list.append(x_start)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma**2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

            imgs.append(img)
            x_start_list.append(x_start)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim=1)
        x_start_list = torch.stack(x_start_list, dim=1)

        ret = self.unnormalize(ret)
        x_start_list = self.unnormalize(x_start_list)
        return ret, x_start_list

    @torch.inference_mode()
    def sample(
        self, batch_size: int = 16, return_all_timesteps: bool = False
    ) -> tuple[Tensor, Tensor]:
        height, width = self.image_size
        channels = self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn(
            (batch_size, channels, height, width),
            return_all_timesteps=return_all_timesteps,
        )

    @torch.inference_mode()
    def interpolate(
        self, x1: Tensor, x2: Tensor, t: Optional[Tensor] = None, lam: float = 0.5
    ) -> Tensor:
        """Interpolates between two images.

        Args:
            x1 (Tensor): First image
            x2 (Tensor): Second Image
            t (Optional[Tensor], optional): timesteps. Defaults to None.
            lam (float, optional): interpolation level. Defaults to 0.5.

        Returns:
            Tensor: The interpolated image
        """
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device=device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc="interpolation sample time step", total=t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    @autocast(enabled=False)
    def q_sample(self, x_start: Tensor, t: Tensor, noise: Optional[Tensor] = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(
        self,
        x_start: Tensor,
        t: Tensor,
        noise: Optional[Tensor] = None,
        offset_noise_strength: Optional[float] = None,
    ) -> Tensor:
        """Calculates the loss for the batch.

        Args:
            x_start (Tensor): x_0
            t (Tensor): timesteps
            noise (Optional[Tensor], optional): Optional noise to apply. Defaults to None.
            offset_noise_strength (Optional[float], optional): strength of noise. Defaults to None.

        Returns:
            Tensor: The loss
        """
        b, c, h, w = x_start.shape

        noise = default(noise, lambda: torch.randn_like(x_start))

        # offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise

        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)

        if offset_noise_strength > 0.0:
            offset_noise = torch.randn(x_start.shape[:2], device=self.device)
            noise += offset_noise_strength * rearrange(offset_noise, "b c -> b c 1 1")

        # noise sample

        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.inference_mode():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step

        model_out = self.model(x, t, x_self_cond)

        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x_start
        elif self.objective == "pred_v":
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f"unknown objective {self.objective}")

        loss = F.mse_loss(model_out, target, reduction="none")
        loss = reduce(loss, "b ... -> b (...)", "mean")

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img: Tensor, *args, **kwargs) -> Tensor:
        """Forward pass for the module.

        Args:
            img (Tensor): The batch of samples

        Returns:
            Tensor: The loss for the batch
        """
        (
            b,
            c,
            h,
            w,
            device,
            img_size,
        ) = (
            *img.shape,
            img.device,
            self.image_size,
        )
        assert (
            h == img_size[0] and w == img_size[1]
        ), f"height and width of image must be {img_size}"
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        return self.p_losses(img, t, *args, **kwargs)
