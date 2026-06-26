"""
ESC-50 environmental audio dataset loader.

Downloads ESC-50 (if not already cached), extracts the WAV clips to disk,
and returns (audio_paths, labels) for use with audio engines and evaluation
tasks, mirroring load_stl10 in image_loader.py.

ESC-50: 2,000 labelled 5-second clips across 50 classes (dog, rain, sea waves,
        crying baby, clock tick, helicopter, ...), 44.1kHz mono.
        Organised into 5 cross-validation folds.

To mirror STL-10's train/test split we use:
  train: folds 1-4  (1,600 clips)
  test:  fold 5     (400 clips)
"""

import csv
import io
import os
import urllib.request
import zipfile
from pathlib import Path
from typing import Literal

# Single zip from the official repo, contains audio/ and meta/esc50.csv
ESC50_URL = "https://github.com/karoldvl/ESC-50/archive/refs/heads/master.zip"

# Default cache location next to the data_utils module
_DEFAULT_ROOT = Path(__file__).parent.parent.parent / "data" / "esc50"

_TRAIN_FOLDS = {1, 2, 3, 4}
_TEST_FOLDS = {5}


def _download_and_extract(root: Path) -> Path:
    """Download the ESC-50 master zip and extract it under root. Returns the
    extracted ESC-50-master directory."""
    extracted = root / "ESC-50-master"
    if (extracted / "meta" / "esc50.csv").exists():
        return extracted

    root.mkdir(parents=True, exist_ok=True)
    print(f"ESC-50: downloading from {ESC50_URL} (~600MB, one-time) ...")
    with urllib.request.urlopen(ESC50_URL) as resp:
        data = resp.read()
    print("  Extracting ...")
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(root)
    return extracted


def load_esc50(
    subset: Literal["train", "test"] = "train",
    root: Path | str | None = None,
    max_per_class: int | None = None,
) -> tuple[list[str], list[int]]:
    """Download ESC-50, return (paths, labels) for the requested split.

    Parameters
    ---
    subset : "train" (folds 1-4) or "test" (fold 5)
    root : directory to store the download and extracted clips
    max_per_class : if set, caps the number of clips per class (useful for
                    quick runs)

    Returns
    ---
    audio_paths : list[str] - absolute paths to WAV files on disk
    labels : list[int] - integer class indices (0-49)
    """
    root = Path(root) if root else _DEFAULT_ROOT
    extracted = _download_and_extract(root)
    audio_dir = extracted / "audio"
    meta_csv = extracted / "meta" / "esc50.csv"

    wanted_folds = _TRAIN_FOLDS if subset == "train" else _TEST_FOLDS

    # Read metadata, filter to the requested folds, optionally cap per class.
    class_counts: dict[int, int] = {}
    audio_paths: list[str] = []
    labels: list[int] = []

    with open(meta_csv, newline="") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: r["filename"])

    for row in rows:
        fold = int(row["fold"])
        if fold not in wanted_folds:
            continue
        label = int(row["target"])
        if max_per_class is not None:
            if class_counts.get(label, 0) >= max_per_class:
                continue
            class_counts[label] = class_counts.get(label, 0) + 1
        audio_paths.append(str(audio_dir / row["filename"]))
        labels.append(label)

    print(f"ESC-50 ({subset}): loaded {len(audio_paths)} clips across "
          f"{len(set(labels))} classes from {audio_dir}")
    return audio_paths, labels
