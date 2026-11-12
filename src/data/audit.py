"""Inspect the downloaded dataset without modifying source images."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

LOGGER = logging.getLogger(__name__)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
CLASS_NAMES = {"NORMAL", "PNEUMONIA"}


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def infer_group_id(path: Path, label: str) -> tuple[str, str]:
    """Infer a conservative filename group key and describe its evidence."""
    stem = path.stem.upper()
    if label == "PNEUMONIA":
        match = re.match(r"^(PERSON\d+)_", stem)
        if match:
            return f"pneumonia:{match.group(1)}", "filename_person_token"
    match = re.match(r"^NORMAL\d*-(IM-\d+)", stem)
    if match:
        return f"normal:{match.group(1)}", "filename_study_token_proxy"
    return f"unparsed:{stem}", "filename_only_unverified"


def inspect_image(path: Path) -> dict[str, Any]:
    """Read basic image metadata or return a record describing the failure."""
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
            mode = image.mode
        return {"width": width, "height": height, "mode": mode, "valid": True, "error": ""}
    except (OSError, UnidentifiedImageError, ValueError) as error:
        return {"width": "", "height": "", "mode": "", "valid": False, "error": str(error)}


def build_manifest(data_root: Path) -> list[dict[str, Any]]:
    """Create one metadata record per supported image under split/class folders."""
    records: list[dict[str, Any]] = []
    for path in sorted(data_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        relative = path.relative_to(data_root)
        parts = relative.parts
        split = parts[0].lower() if len(parts) >= 3 else "unknown"
        label = parts[1].upper() if len(parts) >= 3 else "unknown"
        metadata = inspect_image(path)
        group_id, group_basis = infer_group_id(path, label)
        try:
            digest = sha256_file(path)
        except OSError as error:
            digest = ""
            metadata.update(valid=False, error=f"hash_error: {error}")
        records.append(
            {
                "relative_path": relative.as_posix(),
                "split_original": split,
                "label": label,
                "group_id_inferred": group_id,
                "group_basis": group_basis,
                "sha256": digest,
                "width": metadata["width"],
                "height": metadata["height"],
                "mode": metadata["mode"],
                "valid": metadata["valid"],
                "error": metadata["error"],
            }
        )
    return records


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize counts, labels, invalid files, and cross-partition duplicates."""
    counts: dict[str, dict[str, int]] = {}
    by_hash: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        split = str(record["split_original"])
        label = str(record["label"])
        counts.setdefault(split, {}).setdefault(label, 0)
        counts[split][label] += 1
        digest = str(record["sha256"])
        if digest:
            by_hash.setdefault(digest, []).append(record)
    duplicate_groups = [items for items in by_hash.values() if len(items) > 1]
    cross_split_duplicates = [
        items
        for items in duplicate_groups
        if len({str(item["split_original"]) for item in items}) > 1
    ]
    return {
        "image_count": len(records),
        "valid_image_count": sum(bool(row["valid"]) for row in records),
        "invalid_image_count": sum(not bool(row["valid"]) for row in records),
        "counts_by_original_split_and_class": counts,
        "exact_duplicate_group_count": len(duplicate_groups),
        "cross_split_exact_duplicate_group_count": len(cross_split_duplicates),
        "cross_split_exact_duplicates": [
            [row["relative_path"] for row in group] for group in cross_split_duplicates
        ],
        "unrecognized_labels": sorted({str(row["label"]) for row in records} - CLASS_NAMES),
    }


def write_outputs(records: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    """Write CSV and JSON audit artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "dataset_manifest.csv"
    fields = list(records[0].keys()) if records else []
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(records)
    report = summarize(records)
    (output_dir / "dataset_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def main() -> None:
    """Run the dataset inventory CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/chest_xray"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/data_audit"))
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not arguments.data_root.exists():
        parser.error(f"Dataset directory not found: {arguments.data_root}")
    records = build_manifest(arguments.data_root)
    report = write_outputs(records, arguments.output_dir)
    LOGGER.info("Wrote audit for %d images to %s", len(records), arguments.output_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
