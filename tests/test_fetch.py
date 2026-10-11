"""Tests for `memsnav fetch`, against archives served from the local filesystem."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from memsnav.fetch import ChecksumError, fetch, safe_extract
from memsnav.manifest import ArchiveEntry, Manifest, md5sum


def _zip(path: Path, members: dict[str, bytes]) -> ArchiveEntry:
    with zipfile.ZipFile(path, "w") as bundle:
        for name, data in members.items():
            bundle.writestr(name, data)
    return ArchiveEntry(key=path.name, kind="", size=path.stat().st_size, md5=md5sum(path))


@pytest.fixture
def release(tmp_path):
    served = tmp_path / "served"
    served.mkdir()
    raw = _zip(served / "raw_rec.zip", {"rec/Gyroscope.csv": b"g"})
    raw.kind, raw.recording = "raw", "rec"
    processed = _zip(served / "processed.zip", {"rec.csv": b"t", "segments.json": b"[]"})
    processed.kind = "processed"
    maps = _zip(served / "maps.zip", {"rec_gravity.nc": b"m"})
    maps.kind = "maps"
    manifest = Manifest(version="1", record_id="0", files=[raw, processed, maps])
    return manifest, served.as_uri() + "/{key}"


def test_each_kind_lands_where_the_simulator_looks(release, tmp_path):
    manifest, url = release
    dest = tmp_path / "data"
    fetch(["all"], dest, manifest, url_template=url)
    assert (dest / "raw" / "rec" / "Gyroscope.csv").read_bytes() == b"g"
    # Trajectories and maps share a directory: strapdown-rs finds maps beside the CSV.
    assert (dest / "processed" / "rec.csv").is_file()
    assert (dest / "processed" / "rec_gravity.nc").is_file()


def test_flat_unpacks_into_the_destination(release, tmp_path):
    manifest, url = release
    fetch(["processed", "maps"], tmp_path / "input", manifest, flat=True, url_template=url)
    assert (tmp_path / "input" / "rec.csv").is_file()
    assert (tmp_path / "input" / "rec_gravity.nc").is_file()
    assert not (tmp_path / "input" / "raw").exists()


def test_a_second_fetch_downloads_nothing(release, tmp_path, monkeypatch):
    manifest, url = release
    dest = tmp_path / "data"
    fetch(["processed"], dest, manifest, url_template=url)

    def refuse(*args, **kwargs):
        raise AssertionError("downloaded again")

    monkeypatch.setattr("memsnav.fetch.download", refuse)
    fetch(["processed"], dest, manifest, url_template=url)


def test_a_corrupt_download_is_rejected_and_not_unpacked(release, tmp_path):
    manifest, url = release
    manifest.files[1].md5 = "0" * 32
    dest = tmp_path / "data"
    with pytest.raises(ChecksumError):
        fetch(["processed"], dest, manifest, url_template=url)
    assert not (dest / "processed").exists()
    assert not list((dest / ".memsnav").glob("*.zip*"))


def test_an_unpublished_release_says_what_to_do_instead(tmp_path):
    manifest = Manifest(
        version="1",
        files=[ArchiveEntry(key="processed.zip", kind="processed", size=1, md5="0")],
    )
    with pytest.raises(RuntimeError, match="memsnav preprocess"):
        fetch(["processed"], tmp_path, manifest)


def test_an_archive_cannot_write_outside_the_destination(tmp_path):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escaped.txt", b"x")
    with pytest.raises(ValueError, match="escapes"):
        safe_extract(archive, tmp_path / "target")
    assert not (tmp_path / "escaped.txt").exists()


def test_raw_can_be_narrowed_to_named_recordings(release):
    manifest, _ = release
    assert [e.key for e in manifest.select(["raw"], ["rec"])] == ["raw_rec.zip"]
    with pytest.raises(ValueError, match="nope"):
        manifest.select(["raw"], ["nope"])
