"""Helpers for Google Colab environment detection and data loading."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def is_colab() -> bool:
    """Return True when running inside Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def gpu_available() -> str:
    """Return the GPU device name, or 'CPU' if none."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return "CPU"


def mount_drive(
    drive_data_dir: str = "/content/drive/MyDrive/indabax_data",
    raw_dir: str | Path = "data/raw",
) -> None:
    """Mount Google Drive and copy competition CSVs to data/raw/."""
    if not is_colab():
        logger.warning("mount_drive called outside Colab — skipping.")
        return

    from google.colab import drive  # type: ignore[import-untyped]
    drive.mount("/content/drive")

    src = Path(drive_data_dir)
    dst = Path(raw_dir)
    dst.mkdir(parents=True, exist_ok=True)

    for fname in ("Train.csv", "Test.csv", "SampleSubmission.csv"):
        src_file = src / fname
        if not src_file.exists():
            raise FileNotFoundError(
                f"{src_file} not found in Drive. "
                f"Upload it to {drive_data_dir}/"
            )
        shutil.copy2(str(src_file), str(dst / fname))
        logger.info("Copied %s -> %s", src_file, dst / fname)


def upload_files(raw_dir: str | Path = "data/raw") -> None:
    """Interactive file upload in Colab, then move to data/raw/."""
    if not is_colab():
        logger.warning("upload_files called outside Colab — skipping.")
        return

    from google.colab import files  # type: ignore[import-untyped]

    dst = Path(raw_dir)
    dst.mkdir(parents=True, exist_ok=True)

    uploaded = files.upload()
    for fname, content in uploaded.items():
        target = dst / fname
        target.write_bytes(content)
        logger.info("Uploaded %s -> %s", fname, target)
