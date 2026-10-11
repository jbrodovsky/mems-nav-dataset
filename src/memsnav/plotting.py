"""Plotting helpers used while preprocessing: map-bound padding and the street-map preview.

Moved from strapdown-rs's ``analysis/plotting.py``, which keeps the plots that read filter
output. Latitude and longitude are WGS84 decimal degrees throughout.
"""

import numpy as np
from cartopy import crs as ccrs
from cartopy.io import img_tiles as cimgt
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from pandas import DataFrame


def inflate_bounds(
    x_min: int | float,
    x_max: int | float,
    y_min: int | float,
    y_max: int | float,
    buffer: float = 0.1,
) -> tuple[int | float, int | float, int | float, int | float]:
    """Inflate bounding box by a buffer amount. Intended to be used to provide a margin around the area of interest by
    increasing the bounds by a percentage

    :param x_min: minimum value along the x-axis
    :type x_min: int | float
    :param x_max: maximum value along the x-axis
    :type x_max: int | float
    :param y_min: minimum value along the y-axis
    :type y_min: int | float
    :param y_max: maximum value along the y-axis
    :type y_max: int | float
    :param buffer: Buffer amount. Defaults to 0.1.
    :type buffer: float

    Returns:
        tuple[int | float, int | float, int | float, int | float]: Inflated bounding box.
    """
    x_range = x_max - x_min
    y_range = y_max - y_min
    return (
        x_min - x_range * buffer,
        x_max + x_range * buffer,
        y_min - y_range * buffer,
        y_max + y_range * buffer,
    )


def plot_street_map(
    nav: DataFrame,
    margin=0.01,
    title: str | None = None,
    gps: DataFrame | None = None,
) -> Figure:
    """Render a trajectory on an OpenStreetMap background.

    This function draws the trajectory contained in ``nav`` over OpenStreetMap
    tiles using Cartopy. ``nav`` must be a :class:`pandas.DataFrame` with
    ``latitude`` and ``longitude`` columns (decimal degrees, WGS84). The
    ``nav`` trajectory is plotted in a distinct colour (default ``tab:red``).
    If a ``gps`` DataFrame is supplied it is rendered in black (``GPS/Truth``)
    to make comparison straightforward.

    Returns
    - ``matplotlib.figure.Figure``: the created figure so callers can save or
      further modify it.
    """
    lat_arr = nav["latitude"].to_numpy(dtype=float)
    lon_arr = nav["longitude"].to_numpy(dtype=float)
    nav_color = "tab:red"

    valid_mask = np.isfinite(lat_arr) & np.isfinite(lon_arr)
    if not np.any(valid_mask):
        raise ValueError("No valid latitude/longitude points to plot.")

    lat_clean = lat_arr[valid_mask]
    lon_clean = lon_arr[valid_mask]

    # Define the map extent using only valid coordinates
    lat_min, lat_max = float(np.min(lat_clean)), float(np.max(lat_clean))
    lon_min, lon_max = float(np.min(lon_clean)), float(np.max(lon_clean))

    # Create an OSM tiles instance
    osm_tiles = cimgt.OSM()

    # Create a figure using Cartopy with OpenStreetMap background
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={"projection": ccrs.PlateCarree()})
    ax.set_extent(  # ty:ignore[unresolved-attribute]
        [lon_min - margin, lon_max + margin, lat_min - margin, lat_max + margin],
        crs=ccrs.PlateCarree(),
    )  # ty:ignore[unresolved-attribute]

    # Add the OSM tiles to the map
    ax.add_image(osm_tiles, 12)  # ty:ignore[invalid-argument-type, too-many-positional-arguments]

    # Plot the trajectory (nav) using a distinct color
    ax.plot(
        lon_clean,
        lat_clean,
        "-",
        color=nav_color,
        transform=ccrs.PlateCarree(),
        label="Nav",
    )

    # Optionally plot GPS/Truth in black if provided
    if gps is not None:
        try:
            gps_lat = gps["latitude"].to_numpy(dtype=float)
            gps_lon = gps["longitude"].to_numpy(dtype=float)
            gps_mask = np.isfinite(gps_lat) & np.isfinite(gps_lon)
            if np.any(gps_mask):
                ax.plot(
                    gps_lon[gps_mask],
                    gps_lat[gps_mask],
                    "k--",
                    transform=ccrs.PlateCarree(),
                    label="GPS/Truth",
                )
        except Exception:
            pass

    # Add gridlines with labels
    gl = ax.gridlines(draw_labels=True, alpha=0.2)  # ty:ignore[unresolved-attribute]
    gl.top_labels = False
    gl.right_labels = False

    if title is None:
        ax.set_title("Street Map with Trajectory Points", fontsize=24)
    else:
        ax.set_title(title, fontsize=24)

    ax.legend()
    return fig
