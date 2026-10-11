"""Download a MEMS-Nav release from Zenodo, check it, and unpack it.

Standard library only, so fetching works from a bare `pip install` with no GMT and no
plotting stack in reach of the code path.
"""

from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from memsnav.manifest import (
    KIND_DIRECTORY,
    KINDS,
    ZENODO_FILE_URL,
    ArchiveEntry,
    Manifest,
    md5sum,
)

#: Stamp written beside unpacked files, naming the archive and checksum they came from. A
#: second fetch that finds a matching stamp skips the download.
STAMP_DIRECTORY = ".memsnav"


class ChecksumError(RuntimeError):
    """A downloaded archive does not match the manifest."""


def archive_url(manifest: Manifest, entry: ArchiveEntry, url_template: str) -> str:
    """Where to download one archive from."""
    if manifest.record_id is None and "{record_id}" in url_template:
        raise RuntimeError(
            f"Release {manifest.version} has no Zenodo record yet, so there is nothing to "
            "download. Build the processed data from raw recordings with `memsnav "
            "preprocess` instead, or pass --manifest pointing at a published release."
        )
    return url_template.format(record_id=manifest.record_id, key=entry.key)


def download(url: str, destination: Path, expected_md5: str) -> None:
    """Stream `url` to `destination` and check its MD5 before it takes the final name."""
    partial = destination.with_name(destination.name + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, open(partial, "wb") as handle:  # noqa: S310
        shutil.copyfileobj(response, handle, length=1 << 20)
    actual = md5sum(partial)
    if actual != expected_md5:
        partial.unlink()
        raise ChecksumError(f"{destination.name}: MD5 {actual}, manifest says {expected_md5}")
    partial.replace(destination)


def safe_extract(archive: Path, target: Path) -> list[str]:
    """
    Unpack `archive` into `target`, refusing any member that would land outside it.

    `ZipFile.extractall` already strips `..` and absolute prefixes on CPython, but silently;
    a release archive that tries it is malformed, and that should be loud.

    Returns
    -------
    list[str]
        The member names written.
    """
    target.mkdir(parents=True, exist_ok=True)
    root = target.resolve()
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        for name in names:
            member = PurePosixPath(name)
            if member.is_absolute() or ".." in member.parts:
                raise ValueError(f"{archive.name}: member {name!r} escapes the target directory")
            if not (root / name).resolve().is_relative_to(root):
                raise ValueError(f"{archive.name}: member {name!r} escapes the target directory")
        bundle.extractall(target)
    return names


def fetch(
    kinds: list[str],
    dest: Path,
    manifest: Manifest,
    recordings: list[str] | None = None,
    flat: bool = False,
    keep_archives: bool = False,
    url_template: str = ZENODO_FILE_URL,
) -> None:
    """
    Fetch the chosen archives of a release into `dest`.

    Parameters
    ----------
    kinds : list[str]
        Archive kinds from :data:`memsnav.manifest.KINDS`, or `["all"]`.
    dest : Path
        Root directory. Each kind unpacks into its own sub-directory
        (:data:`memsnav.manifest.KIND_DIRECTORY`) unless `flat`.
    manifest : Manifest
        The release to fetch.
    recordings : list[str], optional
        Narrow `raw` to these recordings.
    flat : bool
        Unpack straight into `dest`. Meant for `processed` + `maps`, which strapdown-rs wants
        in one flat directory.
    keep_archives : bool
        Keep the downloaded zips under `dest/.memsnav/` rather than deleting them once
        unpacked.
    url_template : str
        Download URL with `{record_id}` and `{key}` placeholders.
    """
    if "all" in kinds:
        kinds = list(KINDS)
    entries = manifest.select(kinds, recordings)
    if not entries:
        print(f"Release {manifest.version} lists no archives of kind(s) {kinds}.")
        return

    stamps = dest / STAMP_DIRECTORY
    stamps.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        stamp = stamps / f"{entry.key}.md5"
        if stamp.is_file() and stamp.read_text().strip() == entry.md5:
            print(f"{entry.key}: already unpacked, skipping")
            continue
        archive = stamps / entry.key
        if not (archive.is_file() and md5sum(archive) == entry.md5):
            print(f"{entry.key}: downloading {entry.size / 1e6:,.0f} MB")
            download(archive_url(manifest, entry, url_template), archive, entry.md5)
        target = dest if flat else dest / KIND_DIRECTORY[entry.kind]
        names = safe_extract(archive, target)
        print(f"{entry.key}: unpacked {len(names)} file(s) into {target}")
        stamp.write_text(entry.md5 + "\n")
        if not keep_archives:
            archive.unlink()
