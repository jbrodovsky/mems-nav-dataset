"""Build the archives of a MEMS-Nav release, ready to upload to Zenodo.

What goes in, and what is deliberately kept out, is decided here in code rather than by hand at
upload time, so a re-release cannot quietly drop one of the checks:

* **Volunteers' names and e-mail addresses.** Sensor Logger's study prompt stores them in
  `StudyMetadata.json`. Only the answers listed in :data:`STUDY_ANSWERS_KEPT` survive.
* **Device identifiers.** `Metadata.csv`'s `device id` is a per-install UUID. It is replaced by
  a pseudonym (`device-01`, ...) that is stable across recordings, so "the same phone" stays
  answerable without publishing the identifier.
* **Anything that is not a recording.** Only `*.csv` and `StudyMetadata.json` are archived; a
  stray export zip left inside a recording directory is reported and skipped.
* **Empty recordings are kept, and named.** A directory with no sensor data (only
  `Metadata.csv` and an empty `Annotation.csv`) is still archived: the data article counts
  every recording made, including the one that holds nothing, and its numbers must be
  rebuildable from the release.
* **The synthetic arm.** strapdown-rs replaces the phone's gravity and magnetometer columns
  with simulated ones and marks the directory with `synthetic.json`. Such a directory is
  refused: the release is real sensor data.

Archives are deterministic -- sorted members, fixed timestamps -- so packaging the same inputs
twice gives byte-identical files and MD5s.
"""

from __future__ import annotations

import csv
import io
import json
import shutil
import zipfile
from pathlib import Path

from memsnav.manifest import MANIFEST_NAME, ArchiveEntry, Manifest, md5sum

#: StudyMetadata answers that are published. Everything else -- the participant's name and
#: e-mail -- has its `value` removed; the question itself is kept so the file's shape is.
STUDY_ANSWERS_KEPT = frozenset({"Type Of Vehicle"})

#: Files in a recording directory that are metadata rather than sensor streams.
METADATA_FILES = frozenset({"Metadata.csv", "Annotation.csv", "StudyMetadata.json"})

#: strapdown-rs's provenance file for the synthetic arm (`analysis.synthetic.PROVENANCE_FILE`).
SYNTHETIC_MARKER = "synthetic.json"

MAP_SUFFIXES = ("_gravity.nc", "_magnetic.nc", "_relief.nc")

#: Zip timestamps are pinned so archive bytes depend only on content.
FIXED_DATE_TIME = (1980, 1, 1, 0, 0, 0)

ZIP64_THRESHOLD = 0x7FFFFFFF


def _member(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=FIXED_DATE_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    return info


def _add_file(bundle: zipfile.ZipFile, name: str, source: Path) -> None:
    size = source.stat().st_size
    with (
        open(source, "rb") as src,
        bundle.open(_member(name), "w", force_zip64=size > ZIP64_THRESHOLD) as dst,
    ):
        shutil.copyfileobj(src, dst, length=1 << 20)


def _add_bytes(bundle: zipfile.ZipFile, name: str, data: bytes) -> None:
    bundle.writestr(_member(name), data)


def is_empty_recording(recording: Path) -> bool:
    """True when a recording directory holds no sensor stream with any rows."""
    return not any(
        path.stat().st_size > 0 and path.name not in METADATA_FILES
        for path in recording.glob("*.csv")
    )


def device_pseudonyms(recordings: list[Path]) -> dict[str, str]:
    """Map every `device id` in the recordings' `Metadata.csv` to `device-01`, `device-02`, ...

    Numbered in order of first appearance across recordings sorted by name (that is, by time),
    so the numbering is reproducible.
    """
    pseudonyms: dict[str, str] = {}
    for recording in sorted(recordings):
        metadata = recording / "Metadata.csv"
        if not metadata.is_file():
            continue
        with open(metadata, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                device = (row.get("device id") or "").strip()
                if device and device not in pseudonyms:
                    pseudonyms[device] = f"device-{len(pseudonyms) + 1:02d}"
    return pseudonyms


def anonymize_metadata(text: str, pseudonyms: dict[str, str]) -> str:
    """`Metadata.csv` with each `device id` replaced by its pseudonym."""
    rows = list(csv.reader(io.StringIO(text)))
    if not rows or "device id" not in rows[0]:
        return text
    column = rows[0].index("device id")
    for row in rows[1:]:
        if column < len(row) and row[column].strip():
            row[column] = pseudonyms[row[column].strip()]
    out = io.StringIO()
    csv.writer(out, lineterminator="\n").writerows(rows)
    return out.getvalue()


def anonymize_study(text: str) -> str:
    """`StudyMetadata.json` with every answer not in :data:`STUDY_ANSWERS_KEPT` removed."""
    questions = json.loads(text)
    for question in questions:
        if str(question.get("title", "")).strip() not in STUDY_ANSWERS_KEPT:
            question.pop("value", None)
    return json.dumps(questions, indent=2) + "\n"


def package_recording(recording: Path, out: Path, pseudonyms: dict[str, str]) -> list[str]:
    """
    Write `raw_<recording>.zip` holding `<recording>/<file>` for every published file.

    Returns
    -------
    list[str]
        Files in the recording directory that were left out.
    """
    skipped = []
    archive = out / f"raw_{recording.name}.zip"
    with zipfile.ZipFile(archive, "w", allowZip64=True) as bundle:
        for path in sorted(recording.iterdir()):
            name = f"{recording.name}/{path.name}"
            if path.name == "Metadata.csv":
                text = anonymize_metadata(path.read_text(encoding="utf-8"), pseudonyms)
                _add_bytes(bundle, name, text.encode("utf-8"))
            elif path.name == "StudyMetadata.json":
                text = anonymize_study(path.read_text(encoding="utf-8"))
                _add_bytes(bundle, name, text.encode("utf-8"))
            elif path.is_file() and path.suffix == ".csv":
                _add_file(bundle, name, path)
            else:
                skipped.append(path.name)
    return skipped


def processed_members(processed: Path) -> tuple[list[Path], list[Path]]:
    """The processed directory's trajectory files and the maps that belong to them.

    Only maps named after a written trajectory are released. A split recording's parent maps
    stay behind: preprocessing keeps them only to copy from, and nothing reads them.
    """
    if (processed / SYNTHETIC_MARKER).exists():
        raise ValueError(
            f"{processed} holds strapdown-rs's synthetic arm ({SYNTHETIC_MARKER} is present); "
            "the release is the real sensor data. Point --processed at the real arm."
        )
    if not (processed / "segments.json").is_file():
        raise ValueError(f"{processed} has no segments.json; is it a `memsnav preprocess` output?")
    trajectories = sorted(processed.glob("*.csv"))
    stems = {path.stem for path in trajectories}
    files = trajectories + [processed / "segments.json"]
    files += [
        processed / f"{stem}_street_map.png"
        for stem in sorted(stems)
        if (processed / f"{stem}_street_map.png").is_file()
    ]
    maps = [
        processed / f"{stem}{suffix}"
        for stem in sorted(stems)
        for suffix in MAP_SUFFIXES
        if (processed / f"{stem}{suffix}").is_file()
    ]
    return files, maps


def zip_directory_files(files: list[Path], root: Path, archive: Path) -> None:
    """Zip `files`, stored under their path relative to `root`."""
    with zipfile.ZipFile(archive, "w", allowZip64=True) as bundle:
        for path in sorted(files):
            _add_file(bundle, path.relative_to(root).as_posix(), path)


def package(
    raw: Path,
    processed: Path,
    out: Path,
    version: str,
    benchmark: Path | None = None,
) -> Manifest:
    """
    Build every archive of a release into `out` and write its `MANIFEST.json`.

    Parameters
    ----------
    raw : Path
        Directory of raw recordings, one sub-directory each.
    processed : Path
        `memsnav preprocess --getmaps` output: trajectories, `segments.json`, maps.
    out : Path
        Directory for the archives. Must be empty or absent.
    version : str
        Release version recorded in the manifest.
    benchmark : Path, optional
        A staged directory of benchmark files (scenario configs, per-filter summaries),
        zipped as-is into `benchmark.zip`.
    """
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty")
    out.mkdir(parents=True, exist_ok=True)
    files, maps = processed_members(processed)

    recordings = sorted(path for path in raw.iterdir() if path.is_dir())
    pseudonyms = device_pseudonyms(recordings)

    entries: list[tuple[str, str | None]] = []
    for recording in recordings:
        skipped = package_recording(recording, out, pseudonyms)
        for name in skipped:
            print(f"{recording.name}: left out {name} (not a Sensor Logger stream)")
        entries.append(("raw", recording.name))
        empty = " (no sensor data)" if is_empty_recording(recording) else ""
        print(f"{recording.name}: archived{empty}")

    zip_directory_files(files, processed, out / "processed.zip")
    zip_directory_files(maps, processed, out / "maps.zip")
    archives = [("processed.zip", "processed"), ("maps.zip", "maps")]
    if benchmark is not None:
        staged = [path for path in benchmark.rglob("*") if path.is_file()]
        zip_directory_files(staged, benchmark, out / "benchmark.zip")
        archives.append(("benchmark.zip", "benchmark"))

    manifest = Manifest(version=version)
    for kind, recording in entries:
        key = f"raw_{recording}.zip"
        manifest.files.append(_entry(out / key, kind, recording))
    for key, kind in archives:
        manifest.files.append(_entry(out / key, kind))
    manifest.dump(out / MANIFEST_NAME)
    total = sum(entry.size for entry in manifest.files)
    print(f"Wrote {len(manifest.files)} archives, {total / 1e9:.2f} GB, to {out}")
    return manifest


def _entry(path: Path, kind: str, recording: str | None = None) -> ArchiveEntry:
    return ArchiveEntry(
        key=path.name, kind=kind, size=path.stat().st_size, md5=md5sum(path), recording=recording
    )
