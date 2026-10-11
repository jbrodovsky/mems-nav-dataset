"""The release manifest: which archives make up a MEMS-Nav release, and how to check them.

The same schema serves both ends. `memsnav package` writes one beside the archives it builds,
and the copy bundled in this package (`memsnav/manifest.json`) is that file plus the Zenodo
record it was uploaded to -- which is what `memsnav fetch` downloads from.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from importlib import resources
from pathlib import Path

#: What each archive kind holds. `processed` and `maps` unpack into the same directory, because
#: strapdown-rs looks for `<stem>_gravity.nc` / `<stem>_magnetic.nc` beside each trajectory.
KINDS = ("raw", "processed", "maps", "benchmark")

#: Directory, under a fetch destination, that each kind unpacks into.
KIND_DIRECTORY = {
    "raw": "raw",
    "processed": "processed",
    "maps": "processed",
    "benchmark": "benchmark",
}

MANIFEST_NAME = "MANIFEST.json"

ZENODO_FILE_URL = "https://zenodo.org/records/{record_id}/files/{key}?download=1"


@dataclass
class ArchiveEntry:
    """One file of a release."""

    key: str
    """File name on Zenodo. Flat -- Zenodo keys carry no directories."""

    kind: str
    """One of :data:`KINDS`."""

    size: int
    """Bytes."""

    md5: str
    """Hex digest; Zenodo reports the same checksum for every file it stores."""

    recording: str | None = None
    """For a `raw` archive, the recording it holds."""


@dataclass
class Manifest:
    """A release: its version, where it lives, and its archives."""

    version: str
    record_id: str | None = None
    doi: str | None = None
    concept_doi: str | None = None
    files: list[ArchiveEntry] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path | None = None) -> Manifest:
        """Read a manifest; the one bundled with this package when `path` is None."""
        if path is None:
            text = resources.files("memsnav").joinpath("manifest.json").read_text("utf-8")
        else:
            text = Path(path).read_text("utf-8")
        document = json.loads(text)
        zenodo = document.get("zenodo", {})
        return cls(
            version=document["version"],
            record_id=zenodo.get("record_id"),
            doi=zenodo.get("doi"),
            concept_doi=zenodo.get("concept_doi"),
            files=[ArchiveEntry(**entry) for entry in document.get("files", [])],
        )

    def dump(self, path: Path) -> None:
        """Write the manifest as JSON."""
        document = {
            "dataset": "MEMS-Nav",
            "version": self.version,
            "zenodo": {
                "record_id": self.record_id,
                "doi": self.doi,
                "concept_doi": self.concept_doi,
            },
            "files": [
                {k: v for k, v in asdict(entry).items() if v is not None} for entry in self.files
            ],
        }
        Path(path).write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")

    def select(self, kinds: list[str], recordings: list[str] | None = None) -> list[ArchiveEntry]:
        """The entries of the given kinds, narrowed to `recordings` for `raw`."""
        unknown = set(kinds) - set(KINDS)
        if unknown:
            raise ValueError(f"unknown archive kind(s) {sorted(unknown)}; expected {KINDS}")
        chosen = [entry for entry in self.files if entry.kind in kinds]
        if recordings:
            known = {entry.recording for entry in self.files if entry.kind == "raw"}
            missing = set(recordings) - known
            if missing:
                raise ValueError(f"no raw archive for recording(s) {sorted(missing)}")
            chosen = [
                entry for entry in chosen if entry.kind != "raw" or entry.recording in recordings
            ]
        return chosen


def md5sum(path: Path, chunk: int = 1 << 20) -> str:
    """Hex MD5 of a file, read in chunks so a multi-gigabyte archive never sits in memory."""
    digest = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def verify(directory: Path, manifest: Manifest) -> list[str]:
    """
    Check every archive the manifest lists against the files in `directory`.

    Returns
    -------
    list[str]
        One line per problem; empty when every file is present with the right size and MD5.
    """
    problems = []
    for entry in manifest.files:
        path = Path(directory) / entry.key
        if not path.is_file():
            problems.append(f"{entry.key}: missing")
        elif path.stat().st_size != entry.size:
            problems.append(f"{entry.key}: {path.stat().st_size} bytes, expected {entry.size}")
        elif md5sum(path) != entry.md5:
            problems.append(f"{entry.key}: MD5 mismatch")
    return problems
