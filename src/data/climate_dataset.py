import random
import os

import xarray as xr
from torch.utils.data import IterableDataset


class ClimateDataset(IterableDataset):
    def __init__(
        self, seq_len: int, realizations: list[str], data_dir: str, scenario: str
    ):
        self.seq_len = seq_len
        self.realizations = realizations
        self.data_dir = data_dir
        self.scenario = scenario

        # Load an example realization to get the length of the dataset
        example_realization = self.load_data(self.realizations[0])
        self.length = (len(example_realization.time) - self.seq_len + 1) * len(
            realizations
        )

    def load_data(self, realization: str) -> xr.Dataset:
        """Loads the data from the specified paths and returns it as an xarray Dataset."""
        realization_dir = os.path.join(
            self.data_dir, self.scenario, realization, "*.nc"
        )

        dataset = xr.open_mfdataset(realization_dir, combine="by_coords").load()
        return dataset

    def __iter__(self):
        """ Defines the iterator for the dataset"""

        # We don't want to get the same realizations in the same order
        # every epoch
        random.shuffle(self.realizations)

        for realization in self.realizations:
            dataset = self.load_data(realization)

            # Determine how many samples we can get from the realization
            indices = list(range(len(dataset.time) - self.seq_len + 1))
            random.shuffle(indices)

            # Yield 1 sample at a time
            for index in indices:
                yield dataset.isel(time=slice(index, index + self.seq_len))


if __name__ == "__main__":
    from tqdm import tqdm

    dataset = ClimateDataset(
        28,
        ["r1", "r2"],
        "/research/hutchinson/data/ml_climate/prepared_data/IPSL/",
        "rcp85",
    )

    for thing in tqdm(iter(dataset), total=dataset.length):
        x = thing.samples.as_numpy()
        pass
