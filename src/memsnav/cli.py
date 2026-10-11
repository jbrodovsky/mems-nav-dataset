"""The `memsnav` command line."""

import os

# Street maps are written to files; never open a window.
os.environ.setdefault("MPLBACKEND", "Agg")

import sys
from argparse import ArgumentParser
from pathlib import Path

from memsnav import __version__
from memsnav.manifest import KINDS, MANIFEST_NAME, Manifest, verify


def build_parser() -> ArgumentParser:
    """The full `memsnav` parser, one sub-command per task."""
    parser = ArgumentParser(prog="memsnav", description=__doc__)
    parser.add_argument("--version", action="version", version=f"memsnav {__version__}")
    command = parser.add_subparsers(title="command", dest="command", required=True)

    fetch = command.add_parser("fetch", help="Download and unpack a release from Zenodo.")
    fetch.add_argument(
        "what",
        nargs="+",
        choices=[*KINDS, "all"],
        help="Archive kinds to fetch.",
    )
    fetch.add_argument(
        "--dest",
        type=Path,
        default=Path("data"),
        help="Root to unpack into: raw/, processed/ (trajectories and maps), benchmark/.",
    )
    fetch.add_argument(
        "--flat",
        action="store_true",
        help="Unpack straight into --dest instead of a per-kind sub-directory.",
    )
    fetch.add_argument(
        "--recording",
        action="append",
        help="Fetch only this raw recording (repeatable). Default: all of them.",
    )
    fetch.add_argument("--keep-archives", action="store_true", help="Keep the downloaded zips.")
    fetch.add_argument(
        "--manifest",
        type=Path,
        help="Release manifest to fetch (default: the one bundled with this version).",
    )

    from memsnav.preprocess import add_preprocess_arguments

    preprocess = command.add_parser(
        "preprocess", help="Rebuild processed trajectories from raw recordings."
    )
    add_preprocess_arguments(preprocess)

    package = command.add_parser("package", help="Build a release's archives for upload.")
    package.add_argument("--raw", type=Path, default=Path("data/raw"))
    package.add_argument("--processed", type=Path, required=True)
    package.add_argument("--benchmark", type=Path, help="Staged benchmark directory, zipped as-is.")
    package.add_argument("--out", type=Path, required=True, help="Empty directory for archives.")
    package.add_argument("--release-version", required=True, help="Version for the manifest.")

    check = command.add_parser("verify", help="Check archives against a manifest.")
    check.add_argument("directory", type=Path, help="Directory holding the archives.")
    check.add_argument(
        "--manifest",
        type=Path,
        help=f"Manifest to check against (default: <directory>/{MANIFEST_NAME}).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point of the `memsnav` script."""
    args = build_parser().parse_args(argv)

    if args.command == "fetch":
        from memsnav.fetch import fetch

        fetch(
            args.what,
            args.dest,
            Manifest.load(args.manifest),
            recordings=args.recording,
            flat=args.flat,
            keep_archives=args.keep_archives,
        )
    elif args.command == "preprocess":
        from memsnav.preprocess import preprocess_data

        preprocess_data(args)
    elif args.command == "package":
        from memsnav.release import package

        package(args.raw, args.processed, args.out, args.release_version, args.benchmark)
    elif args.command == "verify":
        manifest = Manifest.load(args.manifest or args.directory / MANIFEST_NAME)
        problems = verify(args.directory, manifest)
        for problem in problems:
            print(problem)
        if problems:
            sys.exit(1)
        print(f"All {len(manifest.files)} archives match.")
