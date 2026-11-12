import pytest

from src.data.make_splits import create_split


def make_record(index: int, partition: str, label: str) -> dict[str, str]:
    return {
        "relative_path": f"{partition}/{label}/image-{index}.jpeg",
        "split_original": partition,
        "label": label,
        "group_id_inferred": f"{label}-{index}",
        "sha256": f"hash-{partition}-{index}",
        "valid": "True",
    }


def test_create_split_preserves_test_and_splits_groups() -> None:
    records = [
        make_record(index, "train", label)
        for label in ("NORMAL", "PNEUMONIA")
        for index in range(20)
    ]
    records.append(make_record(999, "val", "NORMAL"))
    records.extend(make_record(1000 + index, "test", "PNEUMONIA") for index in range(3))

    result = create_split(records, validation_fraction=0.2, seed=42)
    train_groups = {row["group_id_inferred"] for row in result if row["split"] == "train"}
    val_groups = {row["group_id_inferred"] for row in result if row["split"] == "val"}
    test_paths = {row["relative_path"] for row in result if row["split"] == "test"}

    assert not train_groups & val_groups
    assert len(test_paths) == 3
    assert sum(row["split"] == "val" for row in result) >= 1


def test_create_split_rejects_invalid_fraction() -> None:
    with pytest.raises(ValueError, match="validation_fraction"):
        create_split([], validation_fraction=1.0, seed=42)


def test_create_split_repartitions_exact_duplicate_across_official_test() -> None:
    records = [make_record(index, "train", "NORMAL") for index in range(4)]
    records.extend(make_record(index + 10, "train", "PNEUMONIA") for index in range(4))
    duplicate_test = make_record(999, "test", "NORMAL")
    duplicate_test["sha256"] = records[0]["sha256"]
    records.append(duplicate_test)

    result = create_split(records, validation_fraction=0.2, seed=42)
    assert {row["split_strategy"] for row in result} == {"all_data_patient_grouped"}
    duplicate_partitions = {
        row["split"] for row in result if row["sha256"] == duplicate_test["sha256"]
    }
    assert len(duplicate_partitions) == 1


def test_exact_duplicates_are_kept_in_same_development_partition() -> None:
    records = [
        make_record(index, "train", label)
        for label in ("NORMAL", "PNEUMONIA")
        for index in range(10)
    ]
    records[1]["sha256"] = records[0]["sha256"]
    records.extend(
        [make_record(index + 100, "test", "NORMAL") for index in range(3)]
        + [make_record(index + 200, "test", "PNEUMONIA") for index in range(3)]
    )

    result = create_split(records, validation_fraction=0.2, seed=42)
    duplicate_partitions = {row["split"] for row in result if row["sha256"] == records[0]["sha256"]}
    assert len(duplicate_partitions) == 1


def test_auto_guard_repartitions_all_folders_on_patient_overlap() -> None:
    records = [
        make_record(index, "train", label)
        for label in ("NORMAL", "PNEUMONIA")
        for index in range(10)
    ]
    overlap = make_record(999, "test", "PNEUMONIA")
    overlap["group_id_inferred"] = records[-1]["group_id_inferred"]
    records.extend([overlap, make_record(1000, "test", "NORMAL")])

    result = create_split(records, validation_fraction=0.15, seed=42)

    assert {row["split_strategy"] for row in result} == {"all_data_patient_grouped"}
    patient_splits: dict[str, set[str]] = {}
    bootstrap_splits: dict[str, set[str]] = {}
    for row in result:
        patient_splits.setdefault(row["group_id_inferred"], set()).add(row["split"])
        bootstrap_splits.setdefault(row["bootstrap_group_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in patient_splits.values())
    assert all(len(splits) == 1 for splits in bootstrap_splits.values())


def test_preserve_official_test_rejects_group_overlap() -> None:
    train = make_record(0, "train", "NORMAL")
    validation = make_record(1, "train", "PNEUMONIA")
    test = make_record(2, "test", "PNEUMONIA")
    test["group_id_inferred"] = train["group_id_inferred"]
    with pytest.raises(RuntimeError, match="overlap detected"):
        create_split(
            [train, validation, test],
            validation_fraction=0.2,
            seed=42,
            strategy="preserve_official_test",
        )
