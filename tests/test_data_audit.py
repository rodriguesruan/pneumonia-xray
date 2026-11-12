from pathlib import Path

from PIL import Image

from src.data.audit import build_manifest, infer_group_id, summarize


def test_infer_group_id_marks_filename_evidence() -> None:
    group, basis = infer_group_id(Path("person42_bacteria_3.jpeg"), "PNEUMONIA")
    assert group == "pneumonia:PERSON42"
    assert basis == "filename_person_token"


def test_manifest_detects_invalid_file_and_counts_classes(tmp_path: Path) -> None:
    normal = tmp_path / "train" / "NORMAL" / "NORMAL2-IM-42.jpeg"
    pneumonia = tmp_path / "test" / "PNEUMONIA" / "person42_virus_1.jpeg"
    broken = tmp_path / "train" / "PNEUMONIA" / "broken.jpeg"
    for image_path in (normal, pneumonia, broken):
        image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (8, 6), color=100).save(normal)
    Image.new("RGB", (6, 8), color=(50, 50, 50)).save(pneumonia)
    broken.write_text("not an image", encoding="utf-8")

    records = build_manifest(tmp_path)
    report = summarize(records)

    assert len(records) == 3
    assert report["valid_image_count"] == 2
    assert report["invalid_image_count"] == 1
    assert report["cross_split_exact_duplicate_group_count"] == 0


def test_summarize_finds_cross_split_exact_duplicate(tmp_path: Path) -> None:
    shared = tmp_path / "shared.jpeg"
    Image.new("L", (5, 5), color=25).save(shared)
    digest_record = {
        "relative_path": "train/NORMAL/a.jpeg",
        "split_original": "train",
        "label": "NORMAL",
        "valid": True,
    }
    from src.data.audit import sha256_file

    digest = sha256_file(shared)
    report = summarize(
        [
            {**digest_record, "sha256": digest},
            {
                **digest_record,
                "relative_path": "test/NORMAL/a.jpeg",
                "split_original": "test",
                "sha256": digest,
            },
        ]
    )
    assert report["cross_split_exact_duplicate_group_count"] == 1
