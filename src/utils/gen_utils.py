import torch
from torch import Tensor
from tqdm import tqdm
from diffusers import DDPMScheduler
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

    device = next(model.parameters()).device

    # Average across the time dimension, and then repeat along the time dimension
    # To get our average monthly conditioning map
    cond_map = reduce(clean_samples, "b v t h w -> b v 1 h w", "mean").repeat(
        1, 1, clean_samples.shape[-3], 1, 1
    )

    # Sample noise that we'll add to the clean images
    gen_sample = torch.randn_like(clean_samples)

    # Run the diffusion process in reverse
    for i in tqdm(
        range(len(scheduler) - 1, -1, -len(scheduler) // sample_steps),
        "Sampling",
        disable=disable,
    ):
        timestep = torch.tensor([i] * gen_sample.shape[0], device=device)
        output = model(
            gen_sample,
            timestep,
            cond_map=cond_map,
        )

        gen_sample = scheduler.step(output, timestep=i, sample=gen_sample).prev_sample

    return gen_sample
