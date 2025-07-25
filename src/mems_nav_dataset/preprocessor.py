"""
Module for preprocessing data from the [sensor logger](https://github.com/tszheichoi/awesome-sensor-logger) app. Simple CLI interface for pre processing the data in a given directory.
"""

import os
from argparse import ArgumentParser

import pandas as pd
from cartopy import crs as ccrs
from cartopy.io import img_tiles as cimgt
from matplotlib import pyplot as plt
from numpy import float64


def clean_phone_data(dataset_path: str, frequency: int) -> pd.DataFrame:
    """
    Clean the sensor logger app data from the given dataset path.

    Parameters
    ----------
    dataset_path : str
        Path to the dataset file.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with relevant columns.
    """
    assert os.path.exists(dataset_path), f"File {dataset_path} does not exist."
    # Assert the needed .csv files exist
    # assert os.path.exists(os.path.join(dataset_path, "Accelerometer.csv")), "Accelerometer.csv does not exist."
    assert os.path.exists(os.path.join(dataset_path, "Gyroscope.csv")), "Gyroscope.csv does not exist."
    assert os.path.exists(os.path.join(dataset_path, "Magnetometer.csv")), "Magnetometer.csv does not exist."
    assert os.path.exists(os.path.join(dataset_path, "Barometer.csv")), "Barometer.csv does not exist."
    assert os.path.exists(os.path.join(dataset_path, "Gravity.csv")), "Gravity.csv does not exist."
    try:
        assert os.path.exists(os.path.join(dataset_path, "LocationGps.csv")), "LocationGps.csv does not exist."
    except AssertionError:
        assert os.path.exists(os.path.join(dataset_path, "Location.csv")), "Location.csv does not exist."
    assert os.path.exists(os.path.join(dataset_path, "Orientation.csv")), "Orientation.csv does not exist."
    # Read in raw data
    gyroscope = pd.read_csv(os.path.join(dataset_path, "Gyroscope.csv"), index_col=0)
    magnetometer = pd.read_csv(os.path.join(dataset_path, "Magnetometer.csv"), index_col=0)
    barometer = pd.read_csv(os.path.join(dataset_path, "Barometer.csv"), index_col=0)
    gravity = pd.read_csv(os.path.join(dataset_path, "Gravity.csv"), index_col=0)
    orientation = pd.read_csv(os.path.join(dataset_path, "Orientation.csv"), index_col=0)
    try:
        location = pd.read_csv(os.path.join(dataset_path, "LocationGps.csv"), index_col=0)
    except FileNotFoundError:
        location = pd.read_csv(os.path.join(dataset_path, "Location.csv"), index_col=0)
    try:
        accelerometer = pd.read_csv(os.path.join(dataset_path, "TotalAcceleration.csv"), index_col=0)
    except FileNotFoundError as e:
        accelerometer = pd.read_csv(os.path.join(dataset_path, "Accelerometer.csv"), index_col=0)
        accelerometer["x"] += gravity["x"]
        accelerometer["y"] += gravity["y"]
        accelerometer["z"] += gravity["z"]
    # Convert index to datetime
    accelerometer.index = pd.to_datetime(accelerometer.index, utc=True)
    gyroscope.index = pd.to_datetime(gyroscope.index, utc=True)
    magnetometer.index = pd.to_datetime(magnetometer.index, utc=True)
    barometer.index = pd.to_datetime(barometer.index, utc=True)
    gravity.index = pd.to_datetime(gravity.index, utc=True)
    location.index = pd.to_datetime(location.index, utc=True)
    orientation.index = pd.to_datetime(orientation.index, utc=True)
    # Drop "seconds_elapsed" column
    accelerometer.drop(columns=["seconds_elapsed"], inplace=True)
    gyroscope.drop(columns=["seconds_elapsed"], inplace=True)
    magnetometer.drop(columns=["seconds_elapsed"], inplace=True)
    barometer.drop(columns=["seconds_elapsed"], inplace=True)
    gravity.drop(columns=["seconds_elapsed"], inplace=True)
    location.drop(columns=["seconds_elapsed"], inplace=True)
    orientation.drop(columns=["seconds_elapsed"], inplace=True)
    # Rename columns
    magnetometer = magnetometer.rename(columns={"x": "mag_x", "y": "mag_y", "z": "mag_z"})
    accelerometer = accelerometer.rename(columns={"x": "acc_x", "y": "acc_y", "z": "acc_z"})
    gyroscope = gyroscope.rename(columns={"x": "gyro_x", "y": "gyro_y", "z": "gyro_z"})
    gravity = gravity.rename(columns={"x": "grav_x", "y": "grav_y", "z": "grav_z"})

    # Merge dataframes
    data = location.copy()
    data = data.merge(orientation, left_index=True, right_index=True, how="outer")
    data = data.merge(accelerometer, left_index=True, right_index=True, how="outer")
    data = data.merge(gyroscope, left_index=True, right_index=True, how="outer")
    try:
        data = data.merge(magnetometer, left_index=True, right_index=True, how="outer")
    except Exception as e:
        print(f"Error merging magnetometer data: {e}")
    try:
        data = data.merge(barometer, left_index=True, right_index=True, how="outer")
    except Exception as e:
        print(f"Error merging barometer data: {e}")
    try:
        data = data.merge(gravity, left_index=True, right_index=True, how="outer")
    except Exception as e:
        print(f"Error merging gravity data: {e}")
    # Resample to 1Hz
    fqy_str = convert_hz_to_time_str(frequency)
    data = data.resample(fqy_str).mean()
    data = data.dropna()
    # Convert index to datetime
    data.index = pd.to_datetime(data.index, utc=True)
    # Ensure the index is sorted
    data.sort_index(inplace=True)
    return data


def convert_hz_to_time_str(frequency: int) -> str:
    """Convert frequency in Hz to a time string."""
    if frequency <= 0:
        raise ValueError("Frequency must be positive.")
    interval = 1 / int(frequency)
    return f"{interval}s"


def inflate_bounds(
    min_x: float, min_y: float, max_x: float, max_y: float, inflation_percent: float
) -> tuple[float, float, float, float]:
    """
    Used to inflate the cropping bounds for the map section
    """

    # Calculate the width and height of the original bounds
    width = max_x - min_x
    height = max_y - min_y

    # Check if the width or height is near zero and add a small amount
    if width <= 1e-6:
        width = 0.1
    if height <= 1e-6:
        height = 0.1

    # Calculate the amount to inflate based on the percentage
    inflate_x = width * inflation_percent
    inflate_y = height * inflation_percent

    # Calculate the new minimum and maximum coordinates
    new_min_x = min_x - inflate_x
    new_min_y = min_y - inflate_y
    new_max_x = max_x + inflate_x
    new_max_y = max_y + inflate_y

    return new_min_x, new_min_y, new_max_x, new_max_y


def preprocess(args):
    """Preprocess the data based on the provided arguments."""
    base_dir = args.base_dir
    # pictures_dir = os.path.join(base_dir, "pictures")
    output_dir = os.path.join(args.output_dir, f"{int(args.frequency)}Hz")
    frequency = int(args.frequency)
    time_str = convert_hz_to_time_str(args.frequency)
    datasets = os.listdir(base_dir)
    # os.makedirs(os.path.join("data", "cleaned"), exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Preprocessing data from {base_dir} with frequency {frequency} Hz.")
    print(f"Output will be saved to {output_dir}.")

    for dataset in datasets:
        dataset_path = os.path.join(base_dir, dataset)
        cleaned_data = pd.DataFrame()
        if os.path.isdir(dataset_path):
            try:
                cleaned_data = clean_phone_data(dataset_path, frequency)
            except Exception as e:
                print(f"Error processing {dataset}: {e}")
                continue
            cleaned_data.to_csv(os.path.join(output_dir, f"{dataset}.csv"))
            print(f"Cleaned data for {dataset} saved.")
        else:
            print(f"Skipping {dataset}, not a directory.")
            continue
        # Plot the route on a map
        west_lon = cleaned_data["longitude"].min()
        east_lon = cleaned_data["longitude"].max()
        south_lat = cleaned_data["latitude"].min()
        north_lat = cleaned_data["latitude"].max()
        # Inflate bounds by 10% for better visibility
        west_lon, south_lat, east_lon, north_lat = inflate_bounds(west_lon, south_lat, east_lon, north_lat, 0.1)
        # Road map
        extent = [west_lon, east_lon, south_lat, north_lat]
        request = cimgt.GoogleTiles()
        ax = plt.axes(projection=request.crs)
        ax.set_extent(extent)  # type: ignore
        ax.add_image(request, 10)  # type: ignore
        ax.scatter(
            cleaned_data["longitude"],
            cleaned_data["latitude"],
            0.5,
            color="red",
            transform=ccrs.PlateCarree(),
        )
        # ax.set_title(f"Route for {dataset}")
        plt.savefig(os.path.join(args.output_dir, f"{dataset}_route.png"), dpi=300)
        plt.close()
        print(f"Route plot for {dataset} saved.")


def main(base_dir: str = "data/raw", output_dir: str = "data/cleaned", frequency: int = 1) -> None:
    """ "
    "Main function to clean the sensor logger app data from the given base directory."
    """
    parser = ArgumentParser(description="Clean sensor logger app data.")
    parser.add_argument(
        "--base_dir",
        type=str,
        default="data/raw",
        help="Base directory for the sensor logger app data.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Output directory for the cleaned data.",
    )
    parser.add_argument(
        "--frequency",
        type=int,
        default=1,
        help="Frequency in Hz for resampling the data. Default is 1Hz.",
    )
    args = parser.parse_args()
    assert os.path.exists(args.base_dir), f"Base directory {args.base_dir} does not exist."

    preprocess(args)


if __name__ == "__main__":
    main()
