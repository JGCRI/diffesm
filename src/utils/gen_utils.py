import torch
from torch import Tensor
from tqdm import tqdm
from diffusers import DDPMScheduler
from custom_diffusers.continuous_ddpm import ContinuousDDPM
from einops import reduce


@torch.inference_mode()
def generate_samples(
    clean_samples: Tensor,
    scheduler: DDPMScheduler,
    sample_steps: int,
    model: torch.nn.Module,
    disable=False,
):
    """Generate samples from a trained model"""

    # Average across the time dimension, and then repeat along the time dimension
    # To get our average monthly conditioning map
    cond_map = reduce(clean_samples, "b v t h w -> b v 1 h w", "mean").repeat(
        1, 1, clean_samples.shape[-3], 1, 1
    )

    # Sample noise that we'll add to the clean images
    gen_sample = torch.randn_like(clean_samples)

    # set step values
    scheduler.set_timesteps(sample_steps)

    # Run the diffusion process in reverse
    for i in tqdm(
        scheduler.timesteps,
        "Sampling",
        disable=disable,
    ):
        # If we are using a continuous scheduler, convert the timestep to a log_snr
        if isinstance(scheduler, ContinuousDDPM):
            steps = torch.linspace(1.0, 0.0, sample_steps + 1, device=gen_sample.device)
            t = scheduler.log_snr(steps[i]).repeat(clean_samples.shape[0])
        else:
            t = i

        output = model(
            gen_sample,
            t,
            cond_map=cond_map,
        )

        gen_sample = scheduler.step(output, timestep=i, sample=gen_sample).prev_sample

    return gen_sample
