"""Generate validation results and run the final test evaluation once."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data.loaders import build_dataloader
from src.evaluation.calibration import fit_temperature, probabilities_from_logits
from src.evaluation.metrics import (
    bootstrap_confidence_intervals,
    classification_metrics,
    select_threshold_for_sensitivity,
)
from src.models.build import build_model
from src.utils.config import load_yaml
from src.utils.reproducibility import resolve_device

LOGGER = logging.getLogger(__name__)


def checkpoint_digest(checkpoint_path: Path) -> str:
    """Return SHA-256 for binding validation threshold to one model file."""
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split_groups(manifest_path: Path, split: str) -> dict[str, str]:
    """Map image relative paths to split-local bootstrap groups."""
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        return {
            row["relative_path"]: row.get("bootstrap_group_id", row["group_id_inferred"])
            for row in csv.DictReader(handle)
            if row["split"] == split
        }


def predict_split(
    checkpoint_path: Path,
    data_config_path: Path,
    split: str,
    device_name: str,
) -> tuple[list[str], list[int], list[list[float]]]:
    """Load a checkpoint and collect ordered predictions for one split."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    train_config: dict[str, Any] = checkpoint["config"]
    train_config["model"]["pretrained"] = False
    data_config = load_yaml(data_config_path)
    data_loader = build_dataloader(data_config, split, shuffle=False)
    model = build_model(train_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = resolve_device(device_name)
    model.to(device).eval()
    paths: list[str] = []
    labels: list[int] = []
    logits_rows: list[list[float]] = []
    with torch.inference_mode():
        for images, batch_labels, batch_paths in data_loader:
            logits = model(images.to(device, non_blocking=True))
            paths.extend(batch_paths)
            labels.extend(int(value) for value in batch_labels.tolist())
            logits_rows.extend(logits.float().cpu().tolist())
    return paths, labels, logits_rows


def write_predictions(
    output_path: Path,
    paths: list[str],
    labels: list[int],
    probabilities: list[float],
    groups: dict[str, str],
) -> None:
    """Write prediction rows with inferred group identifiers for bootstrap."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["relative_path", "label", "probability", "bootstrap_group_id"]
        )
        writer.writeheader()
        for path, label, probability in zip(paths, labels, probabilities, strict=True):
            writer.writerow(
                {
                    "relative_path": path,
                    "label": label,
                    "probability": probability,
                    "bootstrap_group_id": groups[path],
                }
            )


def run_evaluation(
    checkpoint_path: Path,
    data_config_path: Path,
    evaluation_config_path: Path,
    split: str,
    final_test: bool,
    device_name: str,
) -> Path:
    """Evaluate validation or consume the one-time final test set."""
    checkpoint_path = checkpoint_path.resolve()
    run_dir = checkpoint_path.parent
    output_dir = Path("reports/evaluation") / run_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    digest = checkpoint_digest(checkpoint_path)
    evaluation_config = load_yaml(evaluation_config_path)
    calibration_path = output_dir / "validation_calibration.json"
    if split == "test":
        if not final_test:
            raise ValueError("Test evaluation requires the explicit --final-test flag")
        threshold_path = output_dir / "validation_threshold.json"
        if not threshold_path.exists():
            raise FileNotFoundError("Run validation and freeze a threshold before final test")
        frozen = json.loads(threshold_path.read_text(encoding="utf-8"))
        if frozen["checkpoint_sha256"] != digest:
            raise ValueError(
                "Checkpoint differs from the model used to select the validation threshold"
            )
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        if calibration["checkpoint_sha256"] != digest:
            raise ValueError("Calibration artifact was created with a different checkpoint")
        lock_path = output_dir / "test_evaluation_started.json"
        try:
            with lock_path.open("x", encoding="utf-8") as handle:
                json.dump(
                    {
                        "started_at_utc": datetime.now(UTC).isoformat(),
                        "checkpoint_sha256": digest,
                    },
                    handle,
                )
        except FileExistsError as error:
            raise RuntimeError(
                "The final test has already been started for this experiment"
            ) from error
        threshold = float(frozen["threshold"])
    elif split == "val":
        if final_test:
            raise ValueError("--final-test is only valid with --split test")
        threshold = float(evaluation_config["threshold_selection"]["target_sensitivity"])
    else:
        raise ValueError("split must be either 'val' or 'test'")

    data_config = load_yaml(data_config_path)
    paths, labels, logits = predict_split(checkpoint_path, data_config_path, split, device_name)
    groups_by_path = load_split_groups(Path(data_config["paths"]["split_manifest"]), split)
    groups = np.asarray([groups_by_path[path] for path in paths])
    label_array = np.asarray(labels)
    logits_array = np.asarray(logits)
    positive_index = data_config["labels"]["PNEUMONIA"]
    if split == "val":
        temperature = fit_temperature(
            logits_array,
            label_array,
            evaluation_config["calibration"]["max_iterations"],
        )
        calibration = {"temperature": temperature, "checkpoint_sha256": digest}
        calibration_path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    else:
        temperature = float(calibration["temperature"])
    probability_matrix = probabilities_from_logits(logits_array, temperature)
    probability_array = probability_matrix[:, positive_index]
    if split == "val":
        selection = select_threshold_for_sensitivity(label_array, probability_array, threshold)
        selection["checkpoint_sha256"] = digest
        threshold_path = output_dir / "validation_threshold.json"
        threshold_path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
        threshold = selection["threshold"]

    metrics = classification_metrics(label_array, probability_array, threshold)
    confidence_intervals = bootstrap_confidence_intervals(
        label_array,
        probability_array,
        threshold,
        replicates=evaluation_config["bootstrap_replicates"],
        confidence_level=evaluation_config["confidence_level"],
        seed=evaluation_config["seed"],
        groups=groups,
    )
    result = {
        "split": split,
        "checkpoint_sha256": digest,
        "metrics": metrics,
        "confidence_intervals": confidence_intervals,
    }
    result_path = output_dir / f"{split}_results.json"
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    write_predictions(
        output_dir / f"{split}_predictions.csv",
        paths,
        labels,
        probability_array.tolist(),
        groups_by_path,
    )
    LOGGER.info("Saved %s evaluation to %s", split, result_path)
    return result_path


def main() -> None:
    """Parse CLI options and execute an evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--evaluation-config", type=Path, default=Path("configs/evaluation.yaml"))
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--final-test", action="store_true")
    parser.add_argument("--device", default="auto")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_evaluation(
        arguments.checkpoint,
        arguments.data_config,
        arguments.evaluation_config,
        arguments.split,
        arguments.final_test,
        arguments.device,
    )


if __name__ == "__main__":
    main()
