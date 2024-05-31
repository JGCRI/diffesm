import hydra
import json
import os
from omegaconf import DictConfig

import xarray as xr
from typing_extensions import TypedDict

# Constants
COW_PATH = "/research/hutchinson/"
CLUSTER_PATH = "/cluster/research-groups/hutchinson"


class DATA_DICT(TypedDict):
    load_dir: str
    save_dir: str
    scenario: str
    realizations: dict[str, dict[str, list[str]]]


# How many chunks to split the data into - this will create NUM_CHUNKS separate files to hold the dat
NUM_CHUNKS = 40


def save_dataset(dataset: xr.Dataset, realization: str, save_dir: str, num_chunks):
    """Saves the dataset in chunks to many netCDF4 files for parallel loading later."""

    # Create the save directory if it doesn't already exist
    full_save_dir = os.path.join(save_dir, realization)
    os.makedirs(full_save_dir, exist_ok=True)

    # Delete all the files in the save directory
    for file in os.listdir(full_save_dir):
        os.remove(os.path.join(full_save_dir, file))

    # Determine the number of chunks based on the length of the 'time' dimension
    total_time_points = len(dataset["time"])
    chunk_size = total_time_points // num_chunks

    split_datasets = []
    paths = []
    # 1. Split the dataset into chunks
    for idx in range(num_chunks):
        # 1. Determine the start and end indices for the chunk
        start_idx = idx * chunk_size
        end_idx = start_idx + chunk_size

        # Make sure we cover the remainder if we're on the last chunk
        if idx == num_chunks - 1:
            end_idx = None

        # Slice that chunk from the dataset
        split_datasets.append(dataset.isel(time=slice(start_idx, end_idx)))

        # Save the chunk to an indexed file
        paths.append(os.path.join(full_save_dir, f"chunk_{idx}.nc"))

    xr.save_mfdataset(split_datasets, paths, compute=True)


def process_dataset(dataset: xr.Dataset, start_year: int, end_year: int) -> xr.Dataset:
    """Does any pre-processing of the dataset necessary before we save it to chunks.

    Args:
        dataset (xr.Dataset): Our data

    Returns:
        xr.Dataset: Our processed data
    """

    dataset = dataset.drop_vars(["time_bnds", "lat_bnds", "lon_bnds"], errors="ignore")

    # Drop all the time indices that are greater than 2100
    dataset = dataset.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))

    return dataset


def collect_var_data(path_list: list[str], base_dir: str) -> xr.Dataset:
    """Collects all the data for a given variable and realization."""

    all_data = []
    for path in path_list:
        all_data.append(xr.open_dataset(os.path.join(base_dir, path)))
    return xr.concat(all_data, dim="time").sortby("time").drop("time_bnds")


@hydra.main(version_base=None, config_path="../configs", config_name="prepare_data")
def main(cfg: DictConfig):
    """Goes through each realization in the JSON file and saves the data to chunks.

    Args:
        filename (str): Path to our JSON file
        cluster (bool, optional): Whether we are on the cluster or not. Defaults to False.
    """
    json_path = os.path.join(
        cfg.paths.json_data_dir, cfg.esm, cfg.scenario, "data.json"
    )
    # Open a json file
    with open(json_path) as f:
        data: DATA_DICT = json.load(f)

    # Construct the paths to the load and save directories
    load_dir = data["load_dir"]
    save_dir = os.path.join(
        cfg.paths.data_dir,
        cfg.esm,
        cfg.scenario,
    )

    start_year = cfg.start_year
    end_year = cfg.end_year

    # Iterate through each realization in our JSON file
    for realization, realization_data in data["realizations"].items():
        if realization in ["r1", "r2"]:
            print(realization)
            continue
        # Merge the two variables together
        datasets = [collect_var_data(path_list, load_dir) for path_list in reversed(realization_data.values())]
        datasets[1] = datasets[1].assign_coords({"time" : datasets[0].time})

        dataset = xr.merge(datasets, join="right", compat="override")

        dataset = process_dataset(dataset, start_year, end_year)
        print(f"Finished processing realization {realization}")
        save_dataset(dataset, realization, save_dir, cfg.num_chunks)
        print(f"Finished saving realization {realization}")


if __name__ == "__main__":
    main()
