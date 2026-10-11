"""Tests for `memsnav package`: what a release publishes, and what it must not."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from memsnav.manifest import MANIFEST_NAME, Manifest, verify
from memsnav.release import anonymize_metadata, anonymize_study, package

METADATA_HEADER = (
    "version,device name,recording epoch time,recording time,recording timezone,platform,"
    "appVersion,device id,sensors,sampleRateMs,standardisation"
)

STUDY = [
    {"asked": "Join Study", "title": "Name", "type": "Text", "optional": True, "value": "Ada L"},
    {"asked": "Join Study", "title": "Email", "type": "Text", "optional": True, "value": "a@b.c"},
    {"asked": "Join Study", "title": "Type Of Vehicle ", "type": "Text", "value": "Car"},
]


def _metadata(device: str) -> str:
    return (
        f"{METADATA_HEADER}\n"
        f"3,Pixel 9 Pro,1762709641384,2025-11-09_17-34-01,America/New_York,android,1.48.0,"
        f"{device},Accelerometer|Gyroscope,10|10,true\n"
    )


def _recording(raw: Path, name: str, device: str, study: bool = False) -> Path:
    path = raw / name
    path.mkdir(parents=True)
    (path / "Metadata.csv").write_text(_metadata(device))
    (path / "Annotation.csv").write_text("")
    (path / "Gyroscope.csv").write_text("time,seconds_elapsed,z,y,x\n1,0.0,0.1,0.2,0.3\n")
    if study:
        (path / "StudyMetadata.json").write_text(json.dumps(STUDY))
    return path


def _processed(root: Path, synthetic: bool = False) -> Path:
    processed = root / "processed"
    processed.mkdir()
    for stem in ("2025-01-01_00-00-00", "2025-01-02_00-00-00_A"):
        (processed / f"{stem}.csv").write_text("time,latitude\n")
        (processed / f"{stem}_street_map.png").write_bytes(b"png")
        for kind in ("gravity", "magnetic", "relief"):
            (processed / f"{stem}_{kind}.nc").write_bytes(b"nc")
    # The parent of a split recording: maps kept only to copy from, never released.
    (processed / "2025-01-02_00-00-00_relief.nc").write_bytes(b"parent")
    (processed / "segments.json").write_text("[]")
    if synthetic:
        (processed / "synthetic.json").write_text("{}")
    return processed


@pytest.fixture
def inputs(tmp_path):
    raw = tmp_path / "raw"
    _recording(raw, "2025-01-01_00-00-00", "uuid-b", study=True)
    second = _recording(raw, "2025-01-02_00-00-00", "uuid-a")
    (second / "stray-export.zip").write_bytes(b"not a zip")
    empty = raw / "2025-01-03_00-00-00"
    empty.mkdir()
    (empty / "Metadata.csv").write_text(_metadata("uuid-c"))
    (empty / "Annotation.csv").write_text("")
    return raw, _processed(tmp_path)


def test_names_and_emails_are_removed_and_the_vehicle_kept():
    questions = json.loads(anonymize_study(json.dumps(STUDY)))
    assert [q.get("value") for q in questions] == [None, None, "Car"]
    assert [q["title"] for q in questions] == ["Name", "Email", "Type Of Vehicle "]


def test_device_ids_become_pseudonyms():
    text = anonymize_metadata(_metadata("uuid-b"), {"uuid-b": "device-01"})
    assert "uuid-b" not in text
    assert ",device-01," in text


def test_a_release_publishes_recordings_trajectories_and_their_maps(inputs, tmp_path):
    raw, processed = inputs
    out = tmp_path / "release"
    manifest = package(raw, processed, out, version="9.9.9")

    keys = sorted(entry.key for entry in manifest.files)
    # The empty recording is released too: the article counts it.
    assert keys == [
        "maps.zip",
        "processed.zip",
        "raw_2025-01-01_00-00-00.zip",
        "raw_2025-01-02_00-00-00.zip",
        "raw_2025-01-03_00-00-00.zip",
    ]
    assert verify(out, Manifest.load(out / MANIFEST_NAME)) == []

    with zipfile.ZipFile(out / "raw_2025-01-02_00-00-00.zip") as bundle:
        names = bundle.namelist()
        assert "2025-01-02_00-00-00/stray-export.zip" not in names
        assert "2025-01-02_00-00-00/Gyroscope.csv" in names
        metadata = bundle.read("2025-01-02_00-00-00/Metadata.csv").decode()
    # Numbered by first appearance in time order: uuid-b is in the earlier recording.
    assert ",device-02," in metadata and "uuid-a" not in metadata

    with zipfile.ZipFile(out / "raw_2025-01-03_00-00-00.zip") as bundle:
        assert "device-03" in bundle.read("2025-01-03_00-00-00/Metadata.csv").decode()

    with zipfile.ZipFile(out / "raw_2025-01-01_00-00-00.zip") as bundle:
        study = json.loads(bundle.read("2025-01-01_00-00-00/StudyMetadata.json"))
    assert all("value" not in q for q in study if q["title"] != "Type Of Vehicle ")

    with zipfile.ZipFile(out / "maps.zip") as bundle:
        maps = bundle.namelist()
    assert len(maps) == 6
    assert "2025-01-02_00-00-00_relief.nc" not in maps


def test_packaging_is_byte_for_byte_reproducible(inputs, tmp_path):
    raw, processed = inputs
    first = package(raw, processed, tmp_path / "one", version="1")
    second = package(raw, processed, tmp_path / "two", version="1")
    assert [e.md5 for e in first.files] == [e.md5 for e in second.files]


def test_the_synthetic_arm_is_refused(tmp_path):
    raw = tmp_path / "raw"
    _recording(raw, "2025-01-01_00-00-00", "uuid")
    processed = _processed(tmp_path, synthetic=True)
    with pytest.raises(ValueError, match="synthetic"):
        package(raw, processed, tmp_path / "release", version="1")
