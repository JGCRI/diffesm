import os
import random
from typing import Any

from omegaconf import OmegaConf
import torch
import numpy as np
import xarray as xr
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator

# Constants for the minimum and maximum of our datasets
MIN_MAX_CONSTANTS = {"tas": (-85.0, 60.0), "pr": (0.0, 6.0)}

# Convert from kelvin to celsius and from kg/m^2/s to mm/day
PREPROCESS_FN = {"tas": lambda x: x - 273.15, "pr": lambda x: x * 86400}

# Normalization and Inverse Normalization functions
NORM_FN = {
    "tas": lambda x: x / 20,
    "pr": lambda x: np.log(1 + x),
}
DENORM_FN = {
    "tas": lambda x: x * 20,
    "pr": lambda x: np.exp(x) - 1,
}

# These functions transform the range of the data to [-1, 1]
MIN_MAX_FN = {"tas": lambda x: x}


def min_max_norm(x: Any, min_val: float, max_val: float) -> Any:
    """Normalizes a data array to the range [-1, 1]"""
    return (x - min_val) / (max_val - min_val)


def min_max_denorm(x: Any, min_val: float, max_val: float) -> Any:
    """Inverse normalizes a data array from the range [-1, 1] to [min_val, max_val]"""
    return x * (max_val - min_val) + min_val


def preprocess(ds: xr.DataArray) -> xr.DataArray:
    """Preprocesses a data array"""

    # The name of the variable is contained within the dataarray
    return PREPROCESS_FN[ds.name](ds)


def normalize(ds: xr.DataArray) -> xr.DataArray:
    """Normalizes a data array"""

    # Apply variable specific normalization
    norm = NORM_FN[ds.name](ds)

    # Then apply min-max normalization
    min_val, max_val = MIN_MAX_CONSTANTS[ds.name]
    # norm = min_max_norm(norm, min_val, max_val)
    return norm


def denorm(ds: xr.DataArray) -> xr.DataArray:
    norm = DENORM_FN[ds.name](ds)

    min_val, max_val = MIN_MAX_CONSTANTS[ds.name]
    # norm = min_max_denorm(norm, min_val, max_val)
    return norm


class ClimateDataset(Dataset):
    def __init__(
        self,
        seq_len: int,
        realizations: list[str],
        esm: str,
        data_dir: str,
        scenario: str,
        vars: list[str],
    ):
        self.seq_len = seq_len
        self.realizations = realizations

        self.data_dir = os.path.join(data_dir, esm, scenario)

        # Necessary to convert vars into a Python list
        self.vars = OmegaConf.to_object(vars) if not isinstance(vars, list) else vars

        # Store one dataset (out of memory) as an xarray dataset for metadata
        # Store a different dataset as a torch tensor for speed
        self.xr_data: xr.Dataset
        self.tensor_data: torch.Tensor

        # Load an example realization right off the bat
        self.load_data(self.realizations[0])

    def estimate_num_batches(self, batch_size: int) -> int:
        """Estimates the number of batches in the dataset."""
        return len(self) * len(self.realizations) // batch_size

    def load_data(self, realization: str):
        """Loads the data from the specified paths and returns it as an xarray Dataset."""
        realization_dir = os.path.join(
            self.data_dir, realization, "*.nc"
        )

        # Open up the dataset and make sure it's sorted by time
        dataset = xr.open_mfdataset(realization_dir, combine="by_coords").sortby("time")

        # Only select the variables we are interested in
        dataset = dataset[self.vars]

        # Apply preprocessing and normalization
        self.xr_data = dataset.map(preprocess).map(normalize)
        self.tensor_data = self.convert_xarray_to_tensor(self.xr_data)

    def convert_xarray_to_tensor(self, ds: xr.Dataset) -> torch.Tensor:
        """Generate a tensor of data from an xarray dataset"""

        # Stacks the data variables ('pr', 'tas', ...) into a single dimension
        stacked_ds = ds.to_stacked_array(
            new_dim="var", sample_dims=["time", "lon", "lat"]
        ).transpose("var", "time", "lat", "lon")
 
        # Convert the numpy array to a torch tensor
        tensor_data = torch.tensor(stacked_ds.to_numpy(), dtype=torch.float32)

        return tensor_data

    def convert_tensor_to_xarray(self, tensor: torch.Tensor, coords : xr.DataArray = None) -> xr.Dataset:
        """Generate an xarray dataset from a tensor of data"""

        assert len(tensor.shape) == 4, "Tensor must have shape (var, time, lat, lon)"

        np_data = tensor.cpu().numpy()

        # Convert the numpy array to a dictionary of xr.DataArrays
        # with the same names as the original dataset
        data_vars = {
            var_name: (["time", "lat", "lon"], np_data[i])
            for i, var_name in enumerate(self.xr_data.data_vars.keys())
        }

        # Create the dataset with the same coordinates as the original dataset
        # Note: The original time values are lost and just start at 0 instead
        ds = xr.Dataset(
            data_vars,
            coords={
                "time": np.arange(np_data.shape[1]),
                "lat": np.linspace(-90, 90, np_data.shape[2]),
                "lon": np.linspace(0, 360, np_data.shape[3]),
            },
        ).map(denorm)

        # If we are provided time coords, create a new time coordinate
        if coords is not None:
            ds = ds.assign_coords(coords)
        return ds

    def __len__(self):
        return len(self.xr_data.time) - self.seq_len + 1

    def __getitem__(self, idx: int):
        """Defines how to get a specific index from the dataset"""
        return self.tensor_data[:, idx : idx + self.seq_len]


class ClimateDataLoader:
    def __init__(
        self,
        dataset: ClimateDataset,
        accelerator: Accelerator,
        batch_size: int,
        **dataloader_kwargs: dict[str, Any]
    ):
        self.dataset = dataset
        self.accelerator = accelerator
        self.batch_size = batch_size
        self.dataloader_kwargs = dataloader_kwargs

    def __len__(self):
        return self.dataset.estimate_num_batches(self.batch_size)

    def generate(self) -> torch.Tensor:
        # Iterate through each realization in our dataset
        random.shuffle(self.dataset.realizations)

        for realization in self.dataset.realizations:
            # Load a realization of data into memory
            self.dataset.load_data(realization)

            # Wrap a dataloader around it and generate the data
            dl = self.accelerator.prepare(
                DataLoader(
                    self.dataset, batch_size=self.batch_size, **self.dataloader_kwargs
                )
            )

            for sample in dl:
                yield sample
