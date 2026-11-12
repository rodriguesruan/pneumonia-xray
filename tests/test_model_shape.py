from typing import Any

import pytest


def test_small_cnn_returns_one_logit_per_class() -> None:
    torch: Any = pytest.importorskip("torch")
    model_module: Any = pytest.importorskip("src.models.baseline")
    model = model_module.SmallCNN(num_classes=2, channels=[8, 16], dropout=0.1)
    logits = model(torch.zeros(2, 3, 64, 64))
    assert tuple(logits.shape) == (2, 2)


def test_mobilenet_v3_small_returns_one_logit_per_class() -> None:
    torch: Any = pytest.importorskip("torch")
    model_builder: Any = pytest.importorskip("src.models.build")
    model = model_builder.build_model(
        {
            "model": {
                "name": "mobilenet_v3_small",
                "num_classes": 2,
                "dropout": 0.2,
                "freeze_backbone": True,
                "pretrained": False,
            }
        }
    ).eval()
    with torch.inference_mode():
        logits = model(torch.zeros(2, 3, 96, 96))
    assert tuple(logits.shape) == (2, 2)
