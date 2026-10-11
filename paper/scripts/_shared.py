"""Data root, benchmark loaders and figure style for the data article's build script.

These were borrowed from strapdown-rs's ``papers/anom_combined/scripts`` (``stats.py`` and
``build_figures.py``) while the article lived in that repository. They are copied here, cut
down to what the article uses, and pointed at the release layout instead of strapdown-rs's
output directories, so the article builds from a downloaded release alone.

Data root: the ``MEMSNAV_DATA`` environment variable, defaulting to ``data/`` at the root of
this repository -- the layout ``memsnav fetch all --dest data`` produces:

- ``raw/<recording>/``: the Sensor Logger exports
- ``processed/``: the 10 Hz trajectories, ``segments.json`` and the maps beside them
- ``benchmark/``: strapdown-rs's scores on the processed data (see the release README)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib as mpl
import pandas as pd

# paper/scripts/_shared.py -> repository root is two levels up.
DATA = Path(os.environ.get("MEMSNAV_DATA", Path(__file__).resolve().parents[2] / "data"))
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
BENCHMARK = DATA / "benchmark"

FILTERS = ("ekf", "ukf", "rbpf")
FILTER_LABEL = {"ekf": "EKF", "ukf": "UKF", "rbpf": "RBPF"}
SUMMARY_ROWS = ("mean", "median", "std")

WIDTH_IN = 6.5
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#d9d8d4"

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8.5,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.0,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
    }
)


def performance(filt: str, scenario: str) -> pd.DataFrame:
    """Per-trajectory error statistics of one benchmark run, without the summary rows."""
    path = BENCHMARK / filt / scenario / "performance_summary.csv"
    frame = pd.read_csv(path, index_col=0)
    return frame.drop(index=[r for r in SUMMARY_ROWS if r in frame.index])


def baseline(filt: str, scenario: str) -> dict[str, float]:
    """Horizontal and vertical RMSE statistics of a benchmark run across trajectories."""
    frame = performance(filt, scenario)
    horizontal = frame["RMSE Horizontal Error (m)"]
    vertical = frame["RMSE Vertical Error (m)"]
    return {
        "n": int(frame.shape[0]),
        "h_median": float(horizontal.median()),
        "h_mean": float(horizontal.mean()),
        "h_std": float(horizontal.std()),
        "h_min": float(horizontal.min()),
        "h_max": float(horizontal.max()),
        "v_median": float(vertical.median()),
    }


def geostats() -> dict[str, dict]:
    """Pooled anomaly error statistics of the phone channels, keyed by field."""
    path = BENCHMARK / "geo_stats_pooled.json"
    return {entry["field"]: entry for entry in json.loads(path.read_text())}


def dataset_summary() -> pd.DataFrame:
    """Per-trajectory distance (km) and duration (h)."""
    frame = pd.read_csv(BENCHMARK / "dataset_summary.csv", index_col=0)
    return frame.drop(index=[r for r in SUMMARY_ROWS if r in frame.index])
