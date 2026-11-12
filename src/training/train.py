"""Run one reproducible baseline or transfer-learning experiment."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import shutil
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from src.data.loaders import build_dataloader
from src.models.build import build_model
from src.training.engine import run_epoch
from src.utils.config import load_yaml
from src.utils.reproducibility import resolve_device, seed_everything

LOGGER = logging.getLogger(__name__)


def class_weights_from_manifest(
    split_manifest: Path, labels: dict[str, int], device: torch.device
) -> torch.Tensor:
    """Compute balanced class weights from training rows only."""
    with split_manifest.open(newline="", encoding="utf-8") as handle:
        train_labels = [row["label"] for row in csv.DictReader(handle) if row["split"] == "train"]
    label_indices = [labels[label] for label in train_labels]
    counts = torch.bincount(torch.tensor(label_indices), minlength=len(labels))
    if torch.any(counts == 0):
        raise ValueError("Training split must contain every configured class")
    weights = counts.sum() / (len(labels) * counts.float())
    return weights.to(device)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    best_auc: float,
    stale_epochs: int,
    config: dict[str, Any],
) -> None:
    """Write a resumable checkpoint with model, optimizer, and scheduler state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_auc": best_auc,
            "stale_epochs": stale_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "config": config,
            "versions": {
                "torch": torch.__version__,
                "numpy": np.__version__,
                "scikit_learn": __import__("sklearn").__version__,
            },
        },
        path,
    )


def _limit_batches(loader: Any, maximum: int | None) -> Iterator[Any]:
    """Yield at most the configured number of batches for debug runs."""
    for batch_index, batch in enumerate(loader):
        if maximum is not None and batch_index >= maximum:
            break
        yield batch


def _serializable_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Remove sample-level prediction arrays and non-finite values from logs."""
    return {
        key: value
        for key, value in metrics.items()
        if key not in {"labels", "probabilities"} and np.isfinite(value)
    }


def train(
    config_path: Path,
    data_config_path: Path,
    debug: bool,
    resume_checkpoint: Path | None = None,
) -> Path:
    """Train, validate, early-stop, and save experiment artifacts."""
    config = load_yaml(config_path)
    data_config = load_yaml(data_config_path)
    if debug:
        config["training"]["max_train_batches"] = config["debug"]["max_train_batches"]
        config["training"]["max_val_batches"] = None
        config["training"]["epochs"] = config["debug"]["epochs"]
    seed_everything(data_config["seed"], config["reproducibility"]["deterministic"])
    torch.hub.set_dir(str(Path(".cache/torch/hub").resolve()))
    device = resolve_device(config["device"])
    data_config["loader"]["batch_size"] = config["training"]["batch_size"]
    train_loader = build_dataloader(data_config, "train", shuffle=True)
    val_loader = build_dataloader(data_config, "val", shuffle=False)
    model = build_model(config).to(device)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["training"]["epochs"]
    )
    start_epoch = 0
    best_auc = float("-inf")
    stale_epochs = 0
    if resume_checkpoint is not None:
        checkpoint = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"])
        best_auc = float(checkpoint["best_auc"])
        stale_epochs = int(checkpoint["stale_epochs"])
        random.setstate(checkpoint["python_rng_state"])
        np.random.set_state(checkpoint["numpy_rng_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if torch.cuda.is_available() and checkpoint["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
    class_weights = class_weights_from_manifest(
        Path(data_config["paths"]["split_manifest"]), data_config["labels"], device
    )
    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(config["artifacts"]["run_root"]) / f"{config['model']['name']}-{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    if resume_checkpoint is not None:
        previous_best = resume_checkpoint.parent / "best.pt"
        if previous_best.is_file():
            shutil.copy2(previous_best, run_dir / "best.pt")
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (run_dir / "data_config.yaml").write_text(
        yaml.safe_dump(data_config, sort_keys=False), encoding="utf-8"
    )
    metrics_path = run_dir / "metrics.jsonl"
    tracker_run = None
    if config["tracking"]["enabled"]:
        import wandb

        tracker_run = wandb.init(
            project=config["tracking"]["project"],
            name=run_dir.name,
            config={"train": config, "data": data_config},
        )

    for epoch in range(start_epoch, config["training"]["epochs"]):
        train_metrics = run_epoch(
            model,
            _limit_batches(train_loader, config["training"].get("max_train_batches")),
            loss_function,
            device,
            config["training"]["amp"] and device.type == "cuda",
            config["training"]["monitor_threshold"],
            data_config["labels"]["PNEUMONIA"],
            optimizer,
        )
        val_metrics = run_epoch(
            model,
            _limit_batches(val_loader, config["training"].get("max_val_batches")),
            loss_function,
            device,
            False,
            config["training"]["monitor_threshold"],
            data_config["labels"]["PNEUMONIA"],
        )
        scheduler.step()
        record = {
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": _serializable_metrics(train_metrics),
            "validation": _serializable_metrics(val_metrics),
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        if tracker_run is not None:
            tracker_run.log(record, step=epoch + 1)
        LOGGER.info(
            "epoch=%d train=%s validation=%s",
            epoch + 1,
            record["train"],
            record["validation"],
        )
        current_auc = val_metrics["roc_auc"]
        if np.isfinite(current_auc) and current_auc > best_auc:
            best_auc = current_auc
            stale_epochs = 0
            save_checkpoint(
                run_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                epoch + 1,
                best_auc,
                stale_epochs,
                config,
            )
        else:
            stale_epochs += 1
        save_checkpoint(
            run_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            epoch + 1,
            best_auc,
            stale_epochs,
            config,
        )
        if stale_epochs >= config["training"]["early_stopping_patience"]:
            LOGGER.info(
                "Early stopping after %d epochs without validation AUC improvement",
                stale_epochs,
            )
            break
    if tracker_run is not None:
        tracker_run.finish()
    if not (run_dir / "best.pt").exists():
        raise RuntimeError(
            "Validation ROC-AUC was undefined; verify both classes are present in val"
        )
    return run_dir


def main() -> None:
    """Parse command-line options and launch an experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--resume-checkpoint", type=Path)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_dir = train(
        arguments.config, arguments.data_config, arguments.debug, arguments.resume_checkpoint
    )
    LOGGER.info("Experiment artifacts written to %s", run_dir)


if __name__ == "__main__":
    main()
