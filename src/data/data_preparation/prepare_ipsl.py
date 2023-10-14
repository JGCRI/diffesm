import json
import os
import argparse
from typing_extensions import TypedDict

import xarray as xr
import dask

# Constants
COW_PATH = "/research/hutchinson/"
CLUSTER_PATH = "/cluster/research-groups/hutchinson"

DATA_DICT = TypedDict(
    "DATA_DICT",
    {
        "load_dir": str,
        "save_dir": str,
        "scenario": str,
        "realizations": dict[str, dict[str, list[str]]],
    },
)

# Cut off all data at this date
END_DATE = 2100

# How many chunks to split the data into - this will create NUM_CHUNKS separate files to hold the dat
NUM_CHUNKS = 40


def save_dataset(dataset: xr.Dataset, realization: str, save_dir: str):
    """Saves the dataset in chunks to many netCDF4 files for parallel loading later"""

    # Create the save directory if it doesn't already exist
    full_save_dir = os.path.join(save_dir, realization)
    os.makedirs(full_save_dir, exist_ok=True)

    # Delete all the files in the save directory
    for file in os.listdir(full_save_dir):
        os.remove(os.path.join(full_save_dir, file))

    # Determine the number of chunks based on the length of the 'time' dimension
    total_time_points = len(dataset["time"])
    chunk_size = total_time_points // NUM_CHUNKS

    split_datasets = []
    paths = []
    # 1. Split the dataset into chunks
    for idx in range(NUM_CHUNKS):
        # 1. Determine the start and end indices for the chunk
        start_idx = idx * chunk_size
        end_idx = start_idx + chunk_size

        # Make sure we cover the remainder if we're on the last chunk
        if idx == NUM_CHUNKS - 1:
            end_idx = None

        # Slice that chunk from the dataset
        split_datasets.append(dataset.isel(time=slice(start_idx, end_idx)))

        # Save the chunk to an indexed file
        paths.append(os.path.join(full_save_dir, f"chunk_{idx}.nc"))

    xr.save_mfdataset(split_datasets, paths, compute=True)


def process_dataset(dataset: xr.Dataset) -> xr.Dataset:
    """Does any pre-processing of the dataset necessary before we save it to chunks

    Args:
        dataset (xr.Dataset): Our data

    Returns:
        xr.Dataset: Our processed data
    """

    # Drop all the time indices that are greater than 2100
    dataset = dataset.sel(time=slice(None, f"{END_DATE}-12-31"))

    return dataset


def collect_var_data(path_list: list[str], base_dir: str) -> xr.Dataset:
    """Collects all the data for a given variable and realization."""

    all_data = []
    for path in path_list:
        all_data.append(xr.open_dataset(os.path.join(base_dir, path), engine="netcdf4"))
    return xr.concat(all_data, dim="time")


def merge_variables(var_data_dict: dict[str, xr.Dataset]) -> xr.Dataset:
    """Merges all the variables together for a given realization."""

    # Precipitation coordinates can be buggy so just reassign them from temperature
    var_data_dict["pr"].coords["time"] = var_data_dict["tas"].coords["time"]

    # Separate the variables into a list
    separated_variables = [
        data[var] for var, data in sorted(var_data_dict.items(), key=lambda x: x[0])
    ]

    # Concatenate variables along new dimension and then set the name for that dimension
    merged = xr.concat(separated_variables, dim="var")
    merged["var"] = sorted(list(var_data_dict.keys()))

    # Create a dataset out of the datarray
    merged = xr.Dataset(dict(samples=merged))

    return merged


def main(filename: str, cluster=False):
    """Goes through each realization in the JSON file and saves the data to chunks.

    Args:
        filename (str): Path to our JSON file
        cluster (bool, optional): Whether we are on the cluster or not. Defaults to False.
    """
    # Open a json file
    with open(filename, "r") as f:
        data: DATA_DICT = json.load(f)

    # Construct the paths to the load and save directories
    load_dir = os.path.join(CLUSTER_PATH if cluster else COW_PATH, data["load_dir"])
    save_dir = os.path.join(
        CLUSTER_PATH if cluster else COW_PATH,
        data["save_dir"],
        data["scenario"],
    )

    # Iterate through each realization in our JSON file
    for realization, realization_data in data["realizations"].items():
        # Store the xarray dataset for each
        dataset = merge_variables(
            {
                var: collect_var_data(path_list, load_dir)
                for var, path_list in realization_data.items()
            }
        )

        dataset = process_dataset(dataset)
        print(f"Finished processing realization {realization}")
        save_dataset(dataset, realization, save_dir)
        print(f"Finished saving realization {realization}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a JSON file.")
    parser.add_argument("filename", type=str, help="Path to the JSON file")

    parser.add_argument(
        "--cluster",
        action="store_true",
        help="Flag indicating if we are running on the cluster",
    )

    args = parser.parse_args()

    main(args.filename, args.cluster)
