"""Local educational demo for the pediatric pneumonia classifier."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any
from urllib.request import urlretrieve

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image

# Streamlit Cloud may launch this file with ``app/`` as the working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import build_image_transform
from src.data.loaders import build_dataloader
from src.evaluation.ood import energy_score, fit_energy_threshold
from src.explain.gradcam import create_gradcam
from src.models.build import build_model
from src.utils.config import load_yaml
from src.utils.reproducibility import resolve_device

APP_CONFIG = load_yaml(PROJECT_ROOT / "configs/app.yaml")
CHECKPOINT_SHA256 = "3f8940db6cccbc6b13db06b28e321ab871a6efe5723469b60e2958ef1a35d60f"
PUBLIC_CHECKPOINT_URL = (
    "https://github.com/rodriguesruan/pneumonia-xray/releases/download/" "v0.1.0/best.pt"
)


def checkpoint_digest(checkpoint_path: Path) -> str:
    """Return a content hash used to validate artifacts against the loaded model."""
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@st.cache_resource
def resolve_checkpoint(local_path: str) -> Path:
    """Use a verified local checkpoint or fetch the pinned public release asset."""
    candidate = PROJECT_ROOT / local_path
    if candidate.is_file() and checkpoint_digest(candidate) == CHECKPOINT_SHA256:
        return candidate

    url = os.environ.get("MODEL_CHECKPOINT_URL", PUBLIC_CHECKPOINT_URL)
    target = Path(tempfile.gettempdir()) / "pneumonia_xray_best.pt"
    if target.is_file() and checkpoint_digest(target) == CHECKPOINT_SHA256:
        return target
    target.unlink(missing_ok=True)
    urlretrieve(url, target)
    if checkpoint_digest(target) != CHECKPOINT_SHA256:
        target.unlink(missing_ok=True)
        raise ValueError("O SHA-256 do checkpoint baixado não corresponde ao esperado.")
    return target


def load_validation_artifact(path: Path, checkpoint_sha256: str) -> dict[str, Any] | None:
    """Load a validation artifact only when it belongs to this exact checkpoint."""
    if not path.is_file():
        return None
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("checkpoint_sha256") != checkpoint_sha256:
        return None
    return artifact


@st.cache_resource
def load_model(checkpoint_path: str, device_name: str):
    """Load a trusted local checkpoint and its model architecture."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    train_config: dict[str, Any] = checkpoint["config"]
    train_config["model"]["pretrained"] = False
    model = build_model(train_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = resolve_device(device_name)
    return model.to(device).eval(), train_config, device


@st.cache_data
def validation_energy_threshold(checkpoint_path: str, device_name: str, quantile: float) -> float:
    """Estimate the high-energy alert boundary from validation images."""
    model, _, device = load_model(checkpoint_path, device_name)
    data_config = load_yaml(PROJECT_ROOT / APP_CONFIG["data_config"])
    loader = build_dataloader(data_config, "val", shuffle=False)
    energies = []
    with torch.inference_mode():
        for images, _, _ in loader:
            logits = model(images.to(device, non_blocking=True))
            energies.extend(energy_score(logits).tolist())
    return fit_energy_threshold(np.asarray(energies), quantile)


def mc_dropout_probabilities(
    model: nn.Module, image_tensor: torch.Tensor, passes: int, device: torch.device
) -> np.ndarray:
    """Estimate probability variation by enabling dropout layers at inference."""
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
    predictions = []
    with torch.inference_mode():
        for _ in range(passes):
            logits = model(image_tensor.unsqueeze(0).to(device))
            predictions.append(torch.softmax(logits.float(), dim=1).cpu().numpy()[0])
    model.eval()
    return np.asarray(predictions)


def main() -> None:
    """Render the Streamlit application."""
    st.set_page_config(page_title="Pneumonia X-ray demo", page_icon="🫁", layout="wide")
    st.title("Classificação educacional de raio-X pediátrico")
    st.error(
        "Demonstração educacional. Não é dispositivo médico e não deve orientar diagnóstico, "
        "triagem ou tratamento. Não substitui avaliação profissional."
    )
    st.caption(
        "O modelo foi projetado para um benchmark pediátrico de um centro. Grad-CAM e alertas "
        "de incerteza/OOD são exploratórios e não explicam nem validam um diagnóstico."
    )
    try:
        checkpoint_path = resolve_checkpoint(APP_CONFIG["checkpoint"])
    except (OSError, ValueError) as error:
        st.error(f"Não foi possível carregar o checkpoint verificado: {error}")
        st.stop()
    upload = st.file_uploader("Envie uma imagem JPG ou PNG", type=["jpg", "jpeg", "png"])
    if upload is None:
        st.info("Selecione uma imagem para visualizar a saída exploratória do classificador.")
        st.stop()
    if upload.size > APP_CONFIG["max_upload_mb"] * 1024 * 1024:
        st.error("Arquivo acima do limite configurado.")
        st.stop()
    try:
        image = Image.open(upload).convert("RGB")
    except (OSError, ValueError):
        st.error("Não foi possível ler este arquivo como imagem.")
        st.stop()

    model, train_config, device = load_model(str(checkpoint_path), APP_CONFIG["device"])
    model_digest = checkpoint_digest(checkpoint_path)
    data_config = load_yaml(PROJECT_ROOT / APP_CONFIG["data_config"])
    transform = build_image_transform(data_config, training=False)
    image_size = data_config["image"]["size"]
    image_resized = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    original_rgb = np.asarray(image_resized).astype(np.float32) / 255.0
    image_tensor = transform(image_resized)
    positive_index = data_config["labels"]["PNEUMONIA"]
    predictions = mc_dropout_probabilities(
        model, image_tensor, APP_CONFIG["uncertainty_passes"], device
    )
    calibration_path = PROJECT_ROOT / "app/artifacts/validation_calibration.json"
    calibration = load_validation_artifact(calibration_path, model_digest)
    if calibration is not None:
        temperature = float(calibration["temperature"])
        calibrated_logits = np.log(np.clip(predictions, 1e-12, 1.0)) / temperature
        calibrated_logits -= calibrated_logits.max(axis=1, keepdims=True)
        predictions = np.exp(calibrated_logits)
        predictions /= predictions.sum(axis=1, keepdims=True)
        score_label = "Score calibrado para pneumonia"
    else:
        score_label = "Score bruto para pneumonia (não calibrado)"
    mean_probabilities = predictions.mean(axis=0)
    positive_probability = float(mean_probabilities[positive_index])
    uncertainty = float(predictions[:, positive_index].std())
    with torch.inference_mode():
        logits = model(image_tensor.unsqueeze(0).to(device))
        current_energy = float(energy_score(logits)[0])
    try:
        ood_threshold = validation_energy_threshold(
            str(checkpoint_path), APP_CONFIG["device"], APP_CONFIG["ood_validation_quantile"]
        )
        possible_ood = current_energy > ood_threshold
    except (FileNotFoundError, ValueError, RuntimeError):
        possible_ood = None
        st.info(
            "Alerta OOD não está habilitado na demo pública porque o dataset de validação "
            "não é distribuído com o repositório."
        )

    first_column, second_column = st.columns(2)
    with first_column:
        st.image(image_resized, caption="Imagem recebida", use_container_width=True)
    with second_column:
        st.metric(score_label, f"{positive_probability:.3f}")
        st.metric("Variação MC Dropout", f"{uncertainty:.3f}")
        st.caption(
            "A variação é uma estimativa experimental de incerteza, não uma probabilidade de erro."
        )
        if possible_ood is True:
            st.warning("Energia fora do limite de validação; o resultado pode não ser confiável.")
        elif possible_ood is False:
            st.info(
                "Energia dentro da faixa de validação; isso não garante que a entrada seja adequada."
            )
        threshold_path = PROJECT_ROOT / "app/artifacts/validation_threshold.json"
        threshold_artifact = load_validation_artifact(threshold_path, model_digest)
        if threshold_artifact is not None and calibration is not None:
            frozen_threshold = threshold_artifact["threshold"]
            predicted = positive_probability >= frozen_threshold
            st.write(
                "Saída no threshold selecionado em validação:",
                "PNEUMONIA" if predicted else "NORMAL",
            )
        else:
            st.caption(
                "Threshold/calibração de validação não correspondem ao checkpoint; "
                "exibindo somente o score."
            )

    try:
        cam_image = create_gradcam(
            model,
            train_config["model"]["name"],
            image_tensor.to(device),
            original_rgb,
            positive_index,
        )
    except (AttributeError, ImportError, OSError, RuntimeError):
        cam_image = None
    if cam_image is None:
        st.warning(
            "Grad-CAM indisponível neste ambiente; os demais resultados continuam disponíveis."
        )
    else:
        st.image(
            cam_image,
            caption="Grad-CAM exploratório para a classe pneumonia",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
