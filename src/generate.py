# Standard library imports
import os
from typing import Union
from collections import OrderedDict
from multiprocessing import Queue, Value, Lock
import multiprocessing as mp

# Third party imports
import torch
from torch import Tensor
from torch import nn
from torch.utils.data import DataLoader
import hydra
from omegaconf.omegaconf import DictConfig
from hydra.utils import instantiate
from accelerate import Accelerator
from diffusers import DDPMScheduler
import xarray as xr
from tqdm import tqdm

# Local imports
from data.climate_dataset import ClimateDataset
from utils.gen_utils import generate_samples

Checkpoint = dict[str, Union[int, OrderedDict]]

# Assumes that gen is conditioned on val, and the first realization
# is always reserved for the test set
realization_dict = {"gen": "r2", "val": "r2", "test": "r1"}


def create_batches(
    xr_ds: xr.Dataset,
    dataset: ClimateDataset,
) -> list[xr.Dataset]:
    """Splits the dataset up into batches of size batch_size. This is helpful
    for when we perform multiprocessing, so we can distribute the batches to different
    processes.

    Args:
        xr_ds (xr.Dataset): The xarray dataset that we want to split
        batch_size (int): The size of each batch
        gpu_ids (list[int]): A list of the GPUs we will be utilizing

    Returns:
        list[list[xr.Dataset]]: A list of all the batches, where each batch is a list of xarray datasets
    """
    seq_len = dataset.seq_len

    # Store a list of all batches, and a single batch
    data = []

    # Iterate through every 28 days in the xr dataset
    for i in range(0, len(xr_ds.time), seq_len):
        # Grab a single batch and convert it to tensor
        batch = xr_ds.isel(time=slice(i, i + seq_len))
        tensor_data = dataset.convert_xarray_to_tensor(batch)

        # Append the batch to the list of batches
        data.append((tensor_data, batch.time))

    return data


def custom_collate_fn(
    batches: list[tuple[Tensor, xr.DataArray]]
) -> tuple[Tensor, list[xr.DataArray]]:
    """Collate function for the dataloader. This is necessary because we want to keep track of the time coordinates
    for each batch, so we can convert the generated tensors back into xarray datasets

    Args:
        batches (list[tuple[Tensor, xr.DataArray]]): A list of tuples, where each tuple contains a batch of tensors
        and the corresponding time coordinates

    Returns:
        tuple[Tensor, list[xr.DataArray]]: A tuple containing the stacked tensor batch and a list of time coordinates
    """
    tensor_batch = []
    time_coords = []
    for batch in batches:
        tensor_batch.append(batch[0])
        time_coords.append(batch[1])

    return torch.stack(tensor_batch), time_coords


@hydra.main(version_base=None, config_path="../configs", config_name="generate")
def main(config: DictConfig) -> None:
    # Verify that the save folder exists
    assert os.path.isdir(config.paths.save_dir), "Save directory does not exist"
    assert config.gen_mode in ["gen", "val", "test"], "Invalid gen mode"

    # If we're generating, make sure we have a load path
    if config.gen_mode == "gen":
        assert config.load_path, "Must specify a load path"
        assert os.path.isfile(config.load_path), "Invalid load path"

    # Initialize all necessary objects
    accelerator = Accelerator(**config.accelerator, even_batches=False)

    dataset: ClimateDataset = instantiate(
        config.dataset,
        esm=config.esm,
        scenario=config.scenario,
        data_dir=config.paths.data_dir,
        realizations=[realization_dict[config.gen_mode]],
        vars=[config.variable],
    )
    scheduler: DDPMScheduler = instantiate(config.scheduler)
    scheduler.set_timesteps(config.sample_steps)

    if config.gen_mode == "gen":
        # Load the model from the checkpoint
        chkpt: Checkpoint = torch.load(config.load_path, map_location="cpu")
        model = chkpt["EMA"].eval()
    else:
        model = None
    # Grab the Xarray dataset from the dataset object
    xr_ds = dataset.xr_data.load()

    # Restrict days to the first 28 days of each month and select years
    xr_ds = xr_ds.sel(time=xr_ds.time.dt.day.isin(range(1, 29)))
    xr_ds = xr_ds.sel(time=slice(str(config.start_year), str(config.end_year)))

    batches = create_batches(xr_ds, dataset)

    dataloader = DataLoader(
        batches, batch_size=config.batch_size, collate_fn=custom_collate_fn
    )

    # Prepare the model and dataloader for distributed training
    model, dataloader = accelerator.prepare(model, dataloader)

    gen_samples = []

    for tensor_batch, time_coords in tqdm(
        dataloader, disable=not accelerator.is_main_process
    ):
        if model is not None:
            gen_months = generate_samples(
                tensor_batch,
                scheduler=scheduler,
                sample_steps=config.sample_steps,
                model=model,
                disable=not accelerator.is_main_process,
            )
        else:
            gen_months = tensor_batch

        for i in range(len(gen_months)):
            gen_samples.append(
                dataset.convert_tensor_to_xarray(
                    gen_months[i], time_coords=time_coords[i]
                )
            )

    gen_samples = accelerator.gather_for_metrics(gen_samples)
    gen_samples = xr.concat(gen_samples, "time").drop_vars("height")

    
    if accelerator.is_main_process:
        # Construct the save path
        save_name = f"{config.gen_mode}_{config.save_name + '_' if config.save_name is not None else ''}{config.variable}_{config.start_year}-{config.end_year}.nc"
        save_path = os.path.join(
            config.paths.save_dir, config.esm, config.scenario, save_name
        )
        # Delete the file if it already exists (avoids permission denied errors)
        if os.path.isfile(save_path):
            os.remove(save_path)
        # Save the generated samples
        gen_samples.to_netcdf(save_path)

        os.chmod(save_path, 0o770)


if __name__ == "__main__":
    main()
