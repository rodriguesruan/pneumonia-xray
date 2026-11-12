"""PyTorch dataset and train/evaluation image transforms."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ChestXrayDataset(Dataset[tuple[torch.Tensor, int, str]]):
    """Read images from a generated split manifest on demand."""

    def __init__(
        self,
        data_root: Path,
        manifest_path: Path,
        split: str,
        image_transform: transforms.Compose,
        label_to_index: dict[str, int],
    ) -> None:
        self.data_root = data_root
        self.image_transform = image_transform
        self.label_to_index = label_to_index
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.records = [row for row in rows if row.get("split") == split]
        if not self.records:
            raise ValueError(f"No records found for split '{split}' in {manifest_path}")
        unknown_labels = {row["label"] for row in self.records} - set(label_to_index)
        if unknown_labels:
            raise ValueError(f"Labels missing from label_to_index: {sorted(unknown_labels)}")

    def __len__(self) -> int:
        """Return number of manifest rows in the selected split."""
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        """Load one RGB image and return its tensor, numeric label, and path."""
        record = self.records[index]
        image_path = self.data_root / Path(record["relative_path"])
        with Image.open(image_path) as image:
            image_tensor = self.image_transform(image.convert("RGB"))
        return image_tensor, self.label_to_index[record["label"]], record["relative_path"]


def build_image_transform(config: dict[str, Any], training: bool) -> transforms.Compose:
    """Build configured augmentation and normalization transforms."""
    image_config = config["image"]
    transform_config = config["augmentation"]
    steps: list[Any] = [transforms.Resize((image_config["size"], image_config["size"]))]
    if training:
        steps.append(
            transforms.RandomAffine(
                degrees=transform_config["rotation_degrees"],
                translate=tuple(transform_config["translation_fraction"]),
                scale=tuple(transform_config["scale_range"]),
            )
        )
        steps.append(
            transforms.ColorJitter(
                brightness=transform_config["brightness"],
                contrast=transform_config["contrast"],
            )
        )
        if transform_config["horizontal_flip_probability"] > 0:
            steps.append(
                transforms.RandomHorizontalFlip(p=transform_config["horizontal_flip_probability"])
            )
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=image_config["mean"],
                std=image_config["std"],
            ),
        ]
    )
    return transforms.Compose(steps)
