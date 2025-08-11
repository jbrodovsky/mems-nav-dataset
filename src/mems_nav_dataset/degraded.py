import os
from argparse import ArgumentParser
from glob import glob

import pandas as pd


def run_degraded(
    input_dir: str,
    output_dir: str,
    gps_interval: int,
    gps_accuracy: float,
    gps_spoofing: float,
):
    """
    Create degraded data sets from all .csv files in input_dir using strapdown CLI.
    """
    print(f"Reading data from: {input_dir}")
    print(f"Writing results to: {output_dir}")
    print(f"GPS interval set to: {gps_interval} seconds")
    print(f"GPS accuracy set to: {gps_accuracy} meters")
    print(f"GPS spoofing set to: {gps_spoofing} meters")
    # Determine config string if any gps_* value is not default
    config_str = ""
    if gps_interval != 10:
        config_str += f"{gps_interval}s"
    if gps_accuracy != 1.0:
        config_str += f"_{gps_accuracy}m"
    if gps_spoofing != 0.0:
        config_str += f"_{gps_spoofing}m"
    final_output_dir = (
        os.path.join(output_dir, config_str) if config_str else output_dir
    )
    os.makedirs(final_output_dir, exist_ok=True)
    # Find all .csv files in input_dir recursively
    for input_file in glob(os.path.join(input_dir, "**", "*.csv"), recursive=True):
        base = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(final_output_dir, f"{base}_{config_str}.csv")
        print(f"Processing: {input_file}")
        cmd = f"strapdown --mode closed-loop --input {input_file} --output {output_file} --gps-interval {gps_interval}"
        if gps_accuracy != 1.0:
            cmd += f" --gps-accuracy {gps_accuracy}"
        if gps_spoofing != 0.0:
            cmd += f" --gps-spoofing {gps_spoofing}"
        try:
            ret = os.system(cmd)
            if ret != 0:
                print(
                    f"Skipping {input_file} due to error: strapdown exited with code {ret}"
                )
        except Exception as err:
            print(f"Skipping {input_file} due to error: {err}")
    print("Degraded mechanization data sets created.")


def main():
    parser = ArgumentParser(
        description="Create degraded data sets using strapdown CLI."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing .csv files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for degraded data sets.",
    )
    parser.add_argument("--gps_interval", type=int, required=True, help="GPS interval.")
    parser.add_argument(
        "--gps_accuracy", type=float, required=True, help="GPS accuracy."
    )
    parser.add_argument(
        "--gps_spoofing", type=float, required=True, help="GPS spoofing."
    )
    args = parser.parse_args()
    run_degraded(
        args.input_dir,
        args.output_dir,
        args.gps_interval,
        args.gps_accuracy,
        args.gps_spoofing,
    )


if __name__ == "__main__":
    main()
