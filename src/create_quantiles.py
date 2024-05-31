import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import os

from data.climate_dataset import ClimateDataset, denorm

# Always use the val set for the quantiles
REALIZATION = "r2"

# Always use RCP85 for the scenario
SCENARIO = "rcp85"

# Use 1960-1990 period for computing quantiles
START_YEAR = "1960"
END_YEAR = "1990"


@hydra.main(version_base=None, config_path="../configs", config_name="quantiles.yaml")
def main(cfg: DictConfig):
    # First, create a dataset
    dataset: ClimateDataset = instantiate(
        cfg.dataset,
        data_dir=cfg.paths.data_dir,
        esm=cfg.esm,
        scenario=SCENARIO,
        vars=[cfg.var],
        realizations=[REALIZATION],
    )

    # Extract the xarray dataset and denormalize it
    xr_ds = dataset.xr_data.map(denorm).sel(time=slice(START_YEAR, END_YEAR)).compute()
    breakpoint()
    # Group by each day of the year
    groups = xr_ds.groupby("time.dayofyear")

    quantiles = groups.quantile(q=[0.9, 0.95, 0.99, 0.999], dim="time")

    # Save the quantiles
    save_name = f"{cfg.var}_quantiles.nc"
    save_path = os.path.join(cfg.paths.quantile_dir, cfg.esm, save_name)

    # Delete the file if it already exists (avoids permission denied errors)

    quantiles.to_netcdf(save_path)


if __name__ == "__main__":
    main()
