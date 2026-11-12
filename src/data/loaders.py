"""DataLoader construction from YAML configuration and split manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.data.dataset import ChestXrayDataset, build_image_transform
from src.utils.reproducibility import seed_worker


def build_dataloader(
    config: dict[str, Any], split: str, shuffle: bool = False
) -> DataLoader[tuple[torch.Tensor, int, str]]:
    """Construct a deterministic DataLoader for the selected data split."""
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split '{split}'")
    if split == "test" and shuffle:
        raise ValueError("The held-out test loader cannot be shuffled")
    data_root = Path(config["paths"]["data_root"])
    manifest_path = Path(config["paths"]["split_manifest"])
    is_training = split == "train"
    dataset = ChestXrayDataset(
        data_root=data_root,
        manifest_path=manifest_path,
        split=split,
        image_transform=build_image_transform(config, training=is_training),
        label_to_index=config["labels"],
    )
    generator = torch.Generator()
    generator.manual_seed(config["seed"])
    return DataLoader(
        dataset,
        batch_size=config["loader"]["batch_size"],
        shuffle=shuffle,
        num_workers=config["loader"]["num_workers"],
        pin_memory=config["loader"]["pin_memory"],
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=config["loader"]["num_workers"] > 0,
    )
