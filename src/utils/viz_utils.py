import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import io
import numpy as np


def plot_map(
    ds: xr.Dataset,
    variable: str,
    levels=None,
    cmap=None,
    ax=None,
    fig=None,
    colorbar=False,
):
    """Given some geospatial data and what it is, generates a cartopy
    map

    Args:
        map_data (Tensor): HxW: A tensor containing some values to be mapped
        variable (str): Either 'tas' or 'pr'
    """
    map_data = ds[variable]

    # If no levels are given to function
    if levels is None:
        if variable == "pr":
            levels = [0.1, 0.5, 1.0, 2.0, 4.0, 8.0]

        elif variable == "tas":
            levels = [-60.0, -45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0, 50.0]

    # If no colormap is passed in
    if cmap is None:
        # Set either temp or precip colors
        cmap = "coolwarm" if variable == "tas" else "GnBu"
    else:
        cmap = cmap

    # Creates a matplotlib figure with cartopy and adds coastlines
    if fig is None:
        fig = plt.figure(figsize=(12, 8), dpi=100)

    # If axis is not provided, creat one
    if ax is None:
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson(central_longitude=180))

    dataplot = map_data.plot.imshow(
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        levels=levels,
        cbar_kwargs={"orientation": "horizontal"},
        extend="both",
        interpolation=None,
    )

    ax.coastlines()

    return fig, dataplot


def create_gif(ds: xr.Dataset):
    var_frames = {}
    for var in ds.data_vars.keys():
        # Create a list of frames
        frames = []

        for day in range(ds.time.shape[0]):
            # Create a map of that day
            fig, _ = plot_map(ds.isel(time=day), var)
            plt.tight_layout(pad=0)

            # Saves the figure to a np array
            with io.BytesIO() as buff:
                fig.savefig(buff, format="raw", facecolor="w")
                buff.seek(0)
                data = np.frombuffer(buff.getvalue(), dtype=np.uint8)

            w, h = fig.canvas.get_width_height()
            data = data.reshape((int(h), int(w), -1))

            fig.tight_layout(pad=0)

            # Add the map to the list of frames
            frames.append(data)

            # Close the figure
            plt.close(fig)

        frames = np.stack(frames)
        frames = np.transpose(frames, (0, 3, 1, 2))

        var_frames[var] = frames

    return var_frames
