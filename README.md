# MEMS-Nav

Long road drives recorded with the inertial, magnetic, barometric and GNSS sensors of ordinary
smartphones, for research on navigation when satellite positioning is degraded or absent.

- **29 recordings** from 7 phone models (Google Pixel 6a, 9 Pro, 9 Pro XL; Samsung SM-S921U,
  SM-A146U; Apple iPhone 12 mini, 13 mini), 2023-08 to 2025-11, north-eastern United States,
  mostly at highway speed. Every sensor stream is kept at the rate the phone delivered it,
  typically ~100 Hz inertial and ~1 Hz GNSS.
- **27 processed trajectories** at 10 Hz, cut at inertial dropouts. GNSS fixes stay at their
  native rate and are never interpolated.
- **Map tiles** around each trajectory: relief, free-air gravity anomaly and magnetic anomaly.
- **Reference baselines** from an EKF, a UKF and a Rao-Blackwellized particle filter, run with
  [strapdown-rs](https://github.com/jbrodovsky/strapdown-rs) under full and degraded GNSS.

The data are on Zenodo (DOI pending). This repository holds the code that downloads them,
rebuilds the processed tier from the raw one and packages a release (`memsnav`), together with
the data article (`paper/`). It holds no data itself, so it is small enough to clone, vendor or
depend on.

## Getting the data

```bash
uv tool install git+https://github.com/jbrodovsky/mems-nav-dataset   # or: pip install ...
memsnav fetch processed maps --dest data      # ~1 GB: trajectories + maps in data/processed
memsnav fetch raw --dest data                 # the raw recordings in data/raw/<recording>/
memsnav fetch raw --recording 2025-03-01_16-46-39 --dest data   # just one
memsnav fetch all --dest data                 # everything, as the paper's build expects
```

`fetch` checks every archive against the MD5 in the release manifest
(`src/memsnav/manifest.json`), refuses a mismatch, and skips archives it has already unpacked.
`--flat` unpacks straight into `--dest` instead of a per-kind sub-directory.

## Rebuilding the processed data

```bash
memsnav preprocess -i data/raw -o data/processed -f 10 -b 1.5 --margin-km 10 \
    --getmaps --max-imu-gap-s 5 --min-segment-s 300 --prune
```

These are the parameters of the published release; with them the trajectories and
`segments.json` come out byte-identical to `processed.zip`. Per recording: GNSS from
`LocationGps.csv` (falling back to `Location.csv`), specific force from
`TotalAcceleration.csv` (or `Accelerometer.csv` + `Gravity.csv` on iOS), an outer join on UTC
time, rows before the first fix dropped, 0.1 s bin means, a split wherever the inertial columns
are empty for more than 5 s, and segments under 300 s dropped. `--getmaps` downloads the map
tiles with [PyGMT](https://www.pygmt.org) and needs the `maps` extra
(`pip install "memsnav[maps]"`) and the GMT library; nothing else does.

## Layout

| Zenodo file | Unpacks to | Contents |
|---|---|---|
| `raw_<recording>.zip` | `raw/<recording>/` | Sensor Logger CSV exports, one directory per recording, named by UTC start time |
| `processed.zip` | `processed/` | `<trajectory>.csv` (10 Hz), `<trajectory>_street_map.png`, `segments.json` |
| `maps.zip` | `processed/` | `<trajectory>_{relief,gravity,magnetic}.nc`, beside the trajectory they belong to |
| `benchmark.zip` | `benchmark/` | strapdown-rs scenario configs and per-trajectory error summaries |
| `MANIFEST.json` | -- | size and MD5 of every archive |

A trajectory cut from a recording that had to be split carries a suffix `_A`, `_B`, ... in time
order. `segments.json` records every segment, kept or not, and why. The processed directory is
laid out the way strapdown-rs reads it, with each trajectory's maps beside it, so the
simulator replays every trajectory as is (the specific force is ENU: pass `--enu`).

### Processed columns

`time` (UTC); GNSS: `latitude`, `longitude` (degrees), `altitude` (m), `speed` (m/s), `bearing`
(degrees), `horizontalAccuracy`, `verticalAccuracy`, `speedAccuracy`, `bearingAccuracy`;
attitude: `qw`, `qx`, `qy`, `qz`, `roll`, `pitch`, `yaw`; specific force `acc_*` (m/s²,
including gravity); angular rate `gyro_*` (rad/s); magnetic field `mag_*` (µT, OS-calibrated);
OS gravity `grav_*` (m/s²); `pressure` (hPa) and `relativeAltitude` (m). Body axes are the
phone's. Rows without a GNSS fix carry NaN in the GNSS columns.

## Privacy

The release is built by `memsnav package` (`src/memsnav/release.py`), which removes
volunteers' names and e-mail addresses from `StudyMetadata.json`, replaces the per-install
`device id` in `Metadata.csv` with a stable pseudonym (`device-01`, ...), and archives only
Sensor Logger streams. The location traces themselves are published as recorded.

## Maps

The tiles are cropped from global models served by the GMT remote datasets: SRTM15+ relief
(Tozer et al., 2019), the Sandwell et al. free-air gravity anomaly, and the World Digital
Magnetic Anomaly Map. Cite those sources if you use the tiles.

## Licence and citation

Data and the paper: CC BY 4.0 (`LICENSE-DATA`). Code: MIT (`LICENSE`). Please cite the data
article and the Zenodo record; see `CITATION.cff`.

## Development

```bash
uv sync
uv run ruff check && uv run ruff format --check && uv run pytest -q
```

Building a release:

```bash
memsnav package --raw data/raw --processed data/processed --benchmark <staged dir> \
    --out release/<version> --release-version <version>
memsnav verify release/<version>
```

strapdown-rs's `just dataset-benchmark` stages the benchmark directory.
