"""
Prepares a geonav particle filter dataset from the original cleaned dataset containing the geophysical
measurements and the closed loop INS or degraded INS data.
"""

import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from .earth import calculate_magnetic_field, gravity

# TODO: #103 Fix and verify the gravity anomaly calculations


def prepare_dataset(geo, ins) -> pd.DataFrame:
    # Extract the closed-loop covariance matrix
    n = len(ins)
    covar = [c.split(",") for c in ins["covariance"].values]
    covar = np.array([float(f) for c in covar for f in c])
    covar = covar.reshape((n, 15, 15))
    ins = ins.drop(columns=["covariance"])
    # Get the position certainty, this is principally used in the particle filter initialization
    # but we could potentially use it to give a comparison to the particle filter's performance
    # that accounts for the uncertainty in both the INS and the PF
    ins["latitude_accuracy"] = covar[:, 0, 0]
    ins["longitude_accuracy"] = covar[:, 1, 1]
    ins["altitude_accuracy"] = covar[:, 2, 2]
    ins["velocity_n_accuracy"] = covar[:, 3, 3]
    ins["velocity_e_accuracy"] = covar[:, 4, 4]
    ins["velocity_d_accuracy"] = covar[:, 5, 5]
    # Collect the geophysical measurements
    geo = geo[
        [
            "mag_x",
            "mag_y",
            "mag_z",
            "grav_x",
            "grav_y",
            "grav_z",
            "pressure",
            "relativeAltitude",
        ]
    ].copy()
    # convert geo_data datetime index to seconds from Unix epoch (1970-01-01 00:00:00 UTC)
    unix_epoch = pd.Timestamp("1970-01-01 00:00:00", tz="UTC")
    geo.index = (geo.index - unix_epoch).total_seconds()

    gravity_values = geo[["grav_x", "grav_y", "grav_z"]].values
    gravity_values = np.linalg.norm(gravity_values, axis=1)
    theoretical_gravity = (
        gravity(
            latitude=ins["latitude"].values,
            altitude=np.zeros_like(
                ins["altitude"].values
            ),  # Assuming sea level for theoretical gravity
        )
        # + 3.08e-6 * ins["altitude"].values
    )  # Adjust for altitude (approximation)

    # Calculate the magnetic field in the local-level frame
    magnetic_values = geo[["mag_x", "mag_y", "mag_z"]].values
    magnetic_values = np.linalg.norm(magnetic_values, axis=1)
    theoretical_magnetic = calculate_magnetic_field(
        latitude=ins["latitude"].values,
        longitude=ins["longitude"].values,
        altitude=ins["altitude"].values,
    )
    theoretical_magnetic = np.linalg.norm(theoretical_magnetic, axis=1)
    geo["freeair"] = (gravity_values - theoretical_gravity) * 100000  # Convert to mGal
    geo["magnetic"] = magnetic_values - theoretical_magnetic

    geo_trajectory = pd.merge(
        left=ins,
        right=geo,
        left_index=True,
        right_index=True,
        how="left",
    )

    return geo_trajectory


if __name__ == "__main__":
    parser = ArgumentParser(description="Prepare geonav particle filter dataset")
    parser.add_argument(
        "--geo",
        type=str,
        required=True,
        help="Path to the input directory containing CSV files containing geophysical measurements",
    )
    parser.add_argument(
        "--ins",
        type=str,
        required=True,
        help="Path to the input directory containing CSV files containing closed loop INS or degraded INS data",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to the output CSV file"
    )
    args = parser.parse_args()
    # Ensure the output directory exists
    output_dir = os.path.dirname(args.output)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Output directory created: {output_dir}")
    # get all the files that end in "_degraded.csv" in the current directory
    for root, dirs, files in os.walk(args.ins):
        for file in files:
            if file.endswith("_degraded.csv"):
                print(f"Processing file: {file}")
                base_name = file.split("_degraded.csv")[0]
                print(f"Base name: {base_name}")
                # ins_filename = os.path.join(root, file)
                # base_name = os.path.basename(file)
                # base_name = base_name.split("_")
                # base_name = base_name[0] + "_" + base_name[1]
                # print(f"Using degraded INS data from: {base_name}")
                # Load the dataset
                ins = pd.read_csv(
                    os.path.join(root, file),
                    index_col=0,
                    parse_dates=True,
                    date_format="%S",
                )
                # print(
                #    f"Looking for geophysical data for the same time period here: {os.path.join(args.geo, base_name + '.csv')}"
                # )
                geo = pd.read_csv(
                    os.path.join(args.geo, base_name + ".csv"),
                    index_col=0,
                    parse_dates=True,
                    date_format="%Y-%m-%d %H:%M:%S%z",
                )
                out = prepare_dataset(geo, ins)
                # Save the dataset
                out.to_csv(
                    os.path.join(args.output, base_name + "_geopf.csv"),
                    index_label="timestamp",
                    float_format="%.6f",
                )
                print(f"Prepared dataset saved to {args.output}{base_name}_geopf.csv")
