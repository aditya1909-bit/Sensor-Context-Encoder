"""Shared experiment constants."""

from pathlib import Path

ACTIVITY_NAMES = (
    "walking",
    "walking upstairs",
    "walking downstairs",
    "sitting",
    "standing",
    "laying",
)

SIGNAL_NAMES = (
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
)

VALIDATION_SUBJECTS = frozenset({1, 3, 15, 25, 27})
EXPECTED_TEST_SUBJECTS = frozenset({2, 4, 9, 10, 12, 13, 18, 20, 24})

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
MODEL_REVISION = "c15f933c73438218a2bc078446c513173cc4f06a"
MODEL_HIDDEN_SIZE = 960

PROMPT_PREFIX = (
    "Classify the activity as walking, walking upstairs, walking downstairs, "
    "sitting, standing, or laying.\n\nSensor context: "
)
PROMPT_SUFFIX = "\n\nActivity:"

UCI_URL = (
    "https://archive.ics.uci.edu/static/public/240/"
    "human+activity+recognition+using+smartphones.zip"
)
UCI_ARCHIVE_SHA256 = "c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031"
INNER_ARCHIVE_NAME = "UCI HAR Dataset.zip"
DATASET_DIRECTORY_NAME = "UCI HAR Dataset"


def dataset_root(data_dir: str | Path) -> Path:
    """Return the extracted UCI HAR root under ``data_dir``."""

    return Path(data_dir) / DATASET_DIRECTORY_NAME
