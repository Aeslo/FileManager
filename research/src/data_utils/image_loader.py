"""
STL-10 image dataset loader.

Downloads STL-10 via torchvision (if not already cached), extracts images
to disk as PNG files, and returns (image_paths, labels) for use with
image engines and evaluation tasks.

STL-10: 10 classes (airplane, bird, car, deer, dog, horse, monkey,
        ship, truck, frog), 96×96 RGB images.
  train: 5,000 labelled images
  test:  8,000 labelled images
"""

import os
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

STL10_CLASSES = [
    "airplane", "bird", "car", "deer", "dog",
    "horse", "monkey", "ship", "truck", "frog",
]

# Default cache location next to the data_utils module
_DEFAULT_ROOT = Path(__file__).parent.parent.parent / "data" / "stl10"


def load_stl10(
    subset: Literal["train", "test"] = "train",
    root: Path | str | None = None,
    max_per_class: int | None = None,
) -> tuple[list[str], list[int]]:
    """Download STL-10, save images to disk, return (paths, labels).

    Parameters
    ----------
    subset : "train" or "test"
    root : directory to store raw download and extracted PNGs
    max_per_class : if set, caps the number of images per class (useful for
                    quick runs — e.g. max_per_class=50 gives 500 images total)

    Returns
    -------
    image_paths : list[str]  — absolute paths to PNG files on disk
    labels      : list[int]  — integer class indices (0–9)
    """
    try:
        import torchvision.datasets as tvd
    except ImportError as e:
        raise ImportError("torchvision is required for STL-10. Run: pip install torchvision") from e

    root = Path(root) if root else _DEFAULT_ROOT
    img_dir = root / "images" / subset
    img_dir.mkdir(parents=True, exist_ok=True)

    EXPECTED = {"train": 5000, "test": 8000}

    # Check if already fully extracted
    existing = sorted(img_dir.glob("**/*.png"))
    if len(existing) >= EXPECTED[subset]:
        print(f"STL-10 ({subset}): found {len(existing)} cached images in {img_dir}")
    else:
        if existing:
            print(f"STL-10 ({subset}): partial cache ({len(existing)} images), re-extracting...")
        else:
            print(f"STL-10 ({subset}): downloading and extracting to {img_dir} ...")
        dataset = tvd.STL10(root=str(root / "raw"), split=subset, download=True)
        for idx in range(len(dataset)):
            img_pil, label = dataset[idx]
            class_dir = img_dir / STL10_CLASSES[label]
            class_dir.mkdir(exist_ok=True)
            img_path = class_dir / f"{idx:05d}.png"
            if not img_path.exists():
                img_pil.save(img_path)
        existing = sorted(img_dir.glob("**/*.png"))
        print(f"  Extracted {len(existing)} images.")

    # Build paths + labels, optionally capped per class
    class_counts: dict[int, int] = {}
    image_paths: list[str] = []
    labels: list[int] = []

    for path in existing:
        class_name = path.parent.name
        if class_name not in STL10_CLASSES:
            continue
        label = STL10_CLASSES.index(class_name)
        if max_per_class is not None:
            if class_counts.get(label, 0) >= max_per_class:
                continue
            class_counts[label] = class_counts.get(label, 0) + 1
        image_paths.append(str(path))
        labels.append(label)

    print(f"  Loaded {len(image_paths)} images across {len(set(labels))} classes.")
    return image_paths, labels
