# MEMS-Nav data article (Data in Brief)

The revised MEMS-Nav data paper, retargeted from IJRR to Elsevier *Data in Brief* (fallback: MDPI *Data*).

## Layout

- `main.tex`: `elsarticle` preprint (12pt, author-year natbib), which `\input`s `sections/*.tex` in Data in Brief order
- `sections/`: abstract, specifications table, value of the data, background, data description, methods, limitations, statements
- `tables/`: **generated**, do not edit. `numbers.tex` holds one `\num...` macro for every number quoted in the prose
- `figures/`: **generated**, do not edit
- `scripts/build.py`: builds `tables/`, `figures/`, `scripts/raw_inventory.csv` and `scripts/trajectory_stats.csv`
- `references.bib`, `NOTES.md` (open items for the authors)

## Rebuild

From the repository root:

```bash
# 0. the release, unpacked as raw/, processed/ and benchmark/ under data/
uv run memsnav fetch all --dest data

# 1. tables, figures and in-text numbers (about a minute; reads ~23 GB of raw CSV once)
uv run --with netcdf4 python paper/scripts/build.py

# 2. the PDF
cd paper && latexmk -pdf main.tex
```

`build.py` reads, and never writes, the release under `$MEMSNAV_DATA` (default `data/` at the
repository root):

- `raw/*/`: the Sensor Logger recordings
- `processed/`: the 10 Hz trajectories, `segments.json` and the map tiles
- `benchmark/dataset_summary.csv`, `benchmark/{ekf,ukf,rbpf}/{truth,degraded}/performance_summary.csv`
  and `benchmark/geo_stats_pooled.json`: strapdown-rs's outputs, staged into the release by
  its `just dataset-benchmark` recipe

Before the record is published, point `MEMSNAV_DATA` at a directory laid out the same way
(for instance symlinks to strapdown-rs's `data/raw` and `data/input_real`, and the staged
benchmark). The loaders and the matplotlib style are in `scripts/_shared.py`. The build is
deterministic: two runs give byte-identical tables and figures.

This article lived in strapdown-rs as `papers/mems_nav_data/` until the dataset moved into
this repository.
