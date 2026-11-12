from pathlib import Path

import pytest


def test_batch_shape_and_three_channel_conversion(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from PIL import Image

    from src.data.loaders import build_dataloader
    from src.utils.config import load_yaml

    data_root = tmp_path / "chest_xray"
    manifest_path = tmp_path / "splits.csv"
    rows = []
    for index in range(2):
        relative_path = f"train/NORMAL/image-{index}.jpeg"
        image_path = data_root / relative_path
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("L", (32, 24), color=120).save(image_path)
        rows.append(
            f"{relative_path},train,NORMAL,group-{index},filename_only,{index},32,24,L,True,"
        )
    manifest_path.write_text(
        "relative_path,split_original,label,group_id_inferred,group_basis,sha256,width,"
        "height,mode,valid,error,split\n" + "\n".join(row + ",train" for row in rows),
        encoding="utf-8",
    )
    config_path = Path(__file__).parents[1] / "configs" / "data.yaml"
    config = load_yaml(config_path)
    config["paths"]["data_root"] = str(data_root)
    config["paths"]["split_manifest"] = str(manifest_path)
    config["loader"]["batch_size"] = 2

    batch, labels, paths = next(iter(build_dataloader(config, split="train")))

    assert tuple(batch.shape) == (2, 3, 224, 224)
    assert tuple(labels.shape) == (2,)
    assert len(paths) == 2
