from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import download_data
from sensor_context_encoder.constants import SIGNAL_NAMES, dataset_root


def test_downloads_verifies_and_extracts_nested_archive(
    tmp_path: Path, monkeypatch
) -> None:
    source_directory = tmp_path / "source"
    source_directory.mkdir()
    inner_archive = source_directory / "UCI HAR Dataset.zip"
    with zipfile.ZipFile(inner_archive, "w") as archive:
        for split in ("train", "test"):
            archive.writestr(f"UCI HAR Dataset/{split}/y_{split}.txt", "1\n")
            archive.writestr(f"UCI HAR Dataset/{split}/subject_{split}.txt", "1\n")
            for signal in SIGNAL_NAMES:
                archive.writestr(
                    f"UCI HAR Dataset/{split}/Inertial Signals/{signal}_{split}.txt",
                    "0\n",
                )

    outer_archive = source_directory / "outer.zip"
    with zipfile.ZipFile(outer_archive, "w") as archive:
        archive.write(inner_archive, arcname="UCI HAR Dataset.zip")

    monkeypatch.setattr(download_data, "UCI_URL", "https://example.invalid/uci.zip")
    monkeypatch.setattr(download_data, "UCI_ARCHIVE_SHA256", download_data.sha256(outer_archive))

    def copy_archive(url: str, destination: Path) -> None:
        assert url == download_data.UCI_URL
        shutil.copyfile(outer_archive, destination)

    monkeypatch.setattr(download_data.urllib.request, "urlretrieve", copy_archive)
    data_dir = tmp_path / "data"
    result = download_data.download_dataset(data_dir)
    assert result == dataset_root(data_dir)
    assert download_data.dataset_is_complete(data_dir)
