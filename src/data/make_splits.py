"""Create patient-grouped splits with a guard for overlap in the official test."""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from sklearn.model_selection import StratifiedGroupKFold

from src.utils.config import load_yaml

LOGGER = logging.getLogger(__name__)


def load_manifest(path: Path) -> list[dict[str, str]]:
    """Read a dataset manifest CSV."""
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def create_split(
    records: list[dict[str, str]],
    validation_fraction: float,
    seed: int,
    strategy: str = "auto_guard",
    test_fraction: float = 0.15,
) -> list[dict[str, str]]:
    """Preserve official test only if no inferred patient/duplicate overlap exists."""
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("validation_fraction and test_fraction must be between 0 and 1")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("train, validation, and test fractions must sum to less than one")
    if strategy not in {"auto_guard", "preserve_official_test", "repartition_all"}:
        raise ValueError(f"Unsupported split strategy: {strategy}")
    eligible = [
        row
        for row in records
        if row.get("valid", "").lower() == "true"
        and row.get("label") in {"NORMAL", "PNEUMONIA"}
        and row.get("split_original", "").lower() in {"train", "val", "test"}
    ]
    train_pool = [row for row in eligible if row["split_original"].lower() in {"train", "val"}]
    test_rows = [row for row in eligible if row["split_original"].lower() == "test"]
    if not train_pool or not test_rows:
        raise ValueError("Expected non-empty train/validation pool and official test split")

    test_hashes = {row.get("sha256") for row in test_rows} - {"", None}
    train_hashes = {row.get("sha256") for row in train_pool}
    shared_test_hashes = (test_hashes & train_hashes) - {"", None}
    train_groups = {row["group_id_inferred"] for row in train_pool}
    test_groups = {row["group_id_inferred"] for row in test_rows}
    shared_patient_groups = train_groups & test_groups
    overlap_found = bool(shared_test_hashes or shared_patient_groups)
    if strategy == "preserve_official_test" and overlap_found:
        raise RuntimeError(
            "Potential patient/group or exact-image overlap detected across official "
            "development and test splits; use strategy='repartition_all'"
        )
    if strategy == "repartition_all" or (strategy == "auto_guard" and overlap_found):
        LOGGER.warning(
            "Repartitioning all source folders by inferred patient/study group; "
            "the provided test folder is no longer treated as held out"
        )
        return _repartition_all(eligible, validation_fraction, test_fraction, seed)

    labels = [row["label"] for row in train_pool]
    groups = _connected_group_ids(train_pool)
    folds = max(2, round(1 / validation_fraction))
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    validation_indices: set[int] | None = None
    for _, val_indices in splitter.split(train_pool, labels, groups):
        observed_fraction = len(val_indices) / len(train_pool)
        if validation_indices is None or abs(observed_fraction - validation_fraction) < abs(
            len(validation_indices) / len(train_pool) - validation_fraction
        ):
            validation_indices = set(int(index) for index in val_indices)
    assert validation_indices is not None

    assignments: list[dict[str, str]] = []
    train_group_ids: set[str] = set()
    val_group_ids: set[str] = set()
    for index, row in enumerate(train_pool):
        partition = "val" if index in validation_indices else "train"
        (val_group_ids if partition == "val" else train_group_ids).add(row["group_id_inferred"])
        assignments.append(
            {
                **row,
                "bootstrap_group_id": groups[index],
                "split_strategy": "official_test_guarded",
                "split": partition,
            }
        )
    if train_group_ids & val_group_ids:
        raise RuntimeError("Inferred groups overlap between train and validation")
    test_groups = _connected_group_ids(test_rows)
    assignments.extend(
        {
            **row,
            "bootstrap_group_id": test_groups[index],
            "split_strategy": "official_test_guarded",
            "split": "test",
        }
        for index, row in enumerate(test_rows)
    )

    return assignments


def _select_holdout_indices(records: list[dict[str, str]], fraction: float, seed: int) -> set[int]:
    """Choose the stratified grouped fold closest to a requested sample fraction."""
    groups = _connected_group_ids(records)
    labels = [row["label"] for row in records]
    smallest_class_count = min(Counter(labels).values())
    fold_count = min(max(2, round(1 / fraction)), len(set(groups)), smallest_class_count)
    if fold_count < 2:
        raise ValueError("At least two inferred patient/study groups are required")
    splitter = StratifiedGroupKFold(n_splits=fold_count, shuffle=True, random_state=seed)
    selected: set[int] | None = None
    for _, candidate in splitter.split(records, labels, groups):
        candidate_set = {int(index) for index in candidate}
        if selected is None or abs(len(candidate_set) / len(records) - fraction) < abs(
            len(selected) / len(records) - fraction
        ):
            selected = candidate_set
    if selected is None:
        raise RuntimeError("Could not create a grouped holdout split")
    return selected


def _repartition_all(
    records: list[dict[str, str]], validation_fraction: float, test_fraction: float, seed: int
) -> list[dict[str, str]]:
    """Make new group-disjoint train/validation/test partitions across all folders."""
    test_indices = _select_holdout_indices(records, test_fraction, seed)
    development_rows = [row for index, row in enumerate(records) if index not in test_indices]
    relative_validation_fraction = validation_fraction / (1 - test_fraction)
    validation_indices = _select_holdout_indices(
        development_rows, relative_validation_fraction, seed + 1
    )
    development_groups = _connected_group_ids(development_rows)
    test_rows = [row for index, row in enumerate(records) if index in test_indices]
    test_groups = _connected_group_ids(test_rows)
    assignments: list[dict[str, str]] = []
    for index, row in enumerate(development_rows):
        assignments.append(
            {
                **row,
                "bootstrap_group_id": development_groups[index],
                "split_strategy": "all_data_patient_grouped",
                "split": "val" if index in validation_indices else "train",
            }
        )
    assignments.extend(
        {
            **row,
            "bootstrap_group_id": test_groups[index],
            "split_strategy": "all_data_patient_grouped",
            "split": "test",
        }
        for index, row in enumerate(test_rows)
    )
    return assignments


def _connected_group_ids(records: list[dict[str, str]]) -> list[str]:
    """Join inferred patient/study keys and exact duplicate images transitively.

    Component IDs are derived from stable member keys rather than row indices so that
    independently processed partitions cannot accidentally reuse a bootstrap group ID.
    """
    parent: dict[str, str] = {}

    def find(item: str) -> str:
        parent.setdefault(item, item)
        if parent[item] != item:
            parent[item] = find(parent[item])
        return parent[item]

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    row_tokens: list[str] = []
    for index, row in enumerate(records):
        group_token = f"group:{row['group_id_inferred']}"
        row_token = f"row:{index}"
        union(row_token, group_token)
        digest = row.get("sha256", "")
        if digest:
            union(row_token, f"sha256:{digest}")
        row_tokens.append(row_token)

    component_members: dict[str, set[str]] = {}
    for row_token, row in zip(row_tokens, records, strict=True):
        root = find(row_token)
        members = component_members.setdefault(root, set())
        members.add(f"group:{row['group_id_inferred']}")
        digest = row.get("sha256", "")
        if digest:
            members.add(f"sha256:{digest}")
    component_ids = {
        root: "component:" + hashlib.sha256("\n".join(sorted(members)).encode()).hexdigest()
        for root, members in component_members.items()
    }
    return [component_ids[find(row_token)] for row_token in row_tokens]


def write_split(records: list[dict[str, Any]], output_path: Path) -> None:
    """Write the assigned split manifest."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(records[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    """Run grouped split generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_yaml(arguments.config)
    manifest_path = arguments.manifest or Path(config["paths"]["audit_manifest"])
    output_path = arguments.output or Path(config["paths"]["split_manifest"])
    split_config = config["split"]
    assignments = create_split(
        load_manifest(manifest_path),
        split_config["validation_fraction"],
        config["seed"],
        split_config["strategy"],
        split_config["test_fraction"],
    )
    write_split(assignments, output_path)
    LOGGER.info("Wrote %d split records to %s", len(assignments), output_path)


if __name__ == "__main__":
    main()
