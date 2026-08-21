"""Download and verify the official UCI HAR dataset."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from sensor_context_encoder.constants import (
    INNER_ARCHIVE_NAME,
    SIGNAL_NAMES,
    UCI_ARCHIVE_SHA256,
    UCI_URL,
    dataset_root,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if not target.is_relative_to(destination):
            raise ValueError(f"Unsafe ZIP member path: {member.filename}")
    archive.extractall(destination)


def dataset_is_complete(data_dir: Path) -> bool:
    root = dataset_root(data_dir)
    required = [root / split / f"y_{split}.txt" for split in ("train", "test")]
    required += [root / split / f"subject_{split}.txt" for split in ("train", "test")]
    required += [
        root / split / "Inertial Signals" / f"{signal}_{split}.txt"
        for split in ("train", "test")
        for signal in SIGNAL_NAMES
    ]
    return all(path.is_file() for path in required)


def download_dataset(data_dir: Path, force: bool = False) -> Path:
    if dataset_is_complete(data_dir) and not force:
        print(f"Dataset already available at {dataset_root(data_dir)}")
        return dataset_root(data_dir)

    data_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="uci-har-") as temporary_directory:
        temporary = Path(temporary_directory)
        outer_archive = temporary / "uci_har.zip"
        print(f"Downloading {UCI_URL}")
        urllib.request.urlretrieve(UCI_URL, outer_archive)
        actual_digest = sha256(outer_archive)
        if actual_digest != UCI_ARCHIVE_SHA256:
            raise ValueError(
                f"UCI archive checksum mismatch: expected {UCI_ARCHIVE_SHA256}, "
                f"got {actual_digest}"
            )

        outer_directory = temporary / "outer"
        outer_directory.mkdir()
        with zipfile.ZipFile(outer_archive) as archive:
            safe_extract(archive, outer_directory)

        inner_archive = outer_directory / INNER_ARCHIVE_NAME
        if not inner_archive.is_file():
            raise FileNotFoundError(f"Nested archive {INNER_ARCHIVE_NAME!r} was not found")
        extracted_directory = temporary / "extracted"
        extracted_directory.mkdir()
        with zipfile.ZipFile(inner_archive) as archive:
            safe_extract(archive, extracted_directory)

        source = extracted_directory / "UCI HAR Dataset"
        if not source.is_dir():
            raise FileNotFoundError("Extracted archive does not contain 'UCI HAR Dataset'")
        shutil.copytree(source, dataset_root(data_dir), dirs_exist_ok=True)

    if not dataset_is_complete(data_dir):
        raise RuntimeError("Dataset extraction completed but required inertial files are missing")
    print(f"Dataset ready at {dataset_root(data_dir)}")
    return dataset_root(data_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--force", action="store_true", help="Overwrite an existing extraction")
    args = parser.parse_args()
    download_dataset(args.data_dir, force=args.force)


if __name__ == "__main__":
    main()
