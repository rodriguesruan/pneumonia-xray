"""Create portfolio graphics from the frozen local evaluation artifacts.

The script does not retrain or reevaluate the holdout model. Grad-CAM panels use
six illustrative holdout images and are intended for local portfolio review.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import build_image_transform  # noqa: E402
from src.explain.gradcam import create_gradcam  # noqa: E402
from src.models.build import build_model  # noqa: E402
from src.utils.config import load_yaml  # noqa: E402

RUN_NAME = "mobilenet_v3_small-20261112T171041Z"
EVAL_DIR = ROOT / "reports" / "evaluation" / RUN_NAME
FIG_DIR = ROOT / "reports" / "figures"
CHECKPOINT = ROOT / "runs" / RUN_NAME / "best.pt"
DATA_CONFIG = ROOT / "configs" / "data.yaml"
PALETTE = {
    "navy": "#17324D",
    "blue": "#3977A8",
    "teal": "#2A9D8F",
    "gold": "#E9C46A",
    "coral": "#D96C5F",
    "pale": "#EEF3F7",
    "gray": "#627181",
    "white": "#FFFFFF",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_predictions(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["label"] = int(row["label"])
        row["probability"] = float(row["probability"])
    return rows


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 13,
            "axes.titlesize": 18,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "figure.facecolor": PALETTE["white"],
            "axes.facecolor": PALETTE["white"],
            "savefig.facecolor": PALETTE["white"],
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIG_DIR / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(test: dict) -> None:
    metrics = test["metrics"]
    labels = ["Sensibilidade", "Especificidade", "Precisão", "F1", "ROC-AUC", "PR-AUC"]
    keys = ["sensitivity", "specificity", "precision", "f1", "roc_auc", "pr_auc"]
    values = [100 * metrics[key] for key in keys]
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(16, 8), gridspec_kw={"width_ratios": [0.85, 1.5]}
    )
    fig.suptitle("CNN do zero × modelo final", color=PALETTE["navy"], weight="bold", y=0.98)
    left.set_facecolor(PALETTE["pale"])
    left.axis("off")
    left.text(0.06, 0.84, "BASELINE", fontsize=14, color=PALETTE["gray"], weight="bold")
    left.text(0.06, 0.67, "SmallCNN", fontsize=24, color=PALETTE["navy"], weight="bold")
    left.text(
        0.06,
        0.48,
        "Não treinada nem avaliada\nno holdout. Ganho absoluto\ne relativo: não calculáveis.",
        fontsize=16,
        color=PALETTE["gray"],
        linespacing=1.5,
    )
    left.text(
        0.06,
        0.17,
        "Não comparar com o smoke test de ResNet-50.",
        fontsize=12,
        color=PALETTE["coral"],
    )

    y = np.arange(len(labels))
    right.barh(y, values, color=PALETTE["teal"], height=0.62)
    right.set_yticks(y, labels)
    right.invert_yaxis()
    right.set_xlim(0, 105)
    right.set_xlabel("Resultado no holdout final (%)")
    right.set_title("MobileNetV3 Small · n=870", loc="left", color=PALETTE["navy"], weight="bold")
    right.grid(axis="x", alpha=0.22)
    for row, value in zip(y, values, strict=True):
        right.text(
            value + 1, row, f"{value:.2f}%", va="center", color=PALETTE["navy"], weight="bold"
        )
    fig.text(
        0.51,
        0.02,
        "Comparação de baseline não foi executada neste ciclo; "
        "o gráfico não inventa uma referência.",
        ha="center",
        fontsize=11,
        color=PALETTE["gray"],
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    save(fig, "01_baseline_vs_modelo_final.png")


def plot_confusion(test: dict) -> None:
    m = test["metrics"]
    matrix = np.array(
        [[m["true_negative"], m["false_positive"]], [m["false_negative"], m["true_positive"]]]
    )
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], ["NORMAL", "PNEUMONIA"])
    ax.set_yticks([0, 1], ["NORMAL", "PNEUMONIA"])
    ax.set_xlabel("Predição do modelo")
    ax.set_ylabel("Classe real")
    ax.set_title(
        "Matriz de confusão · holdout agrupado", color=PALETTE["navy"], weight="bold", pad=18
    )
    for i in range(2):
        for j in range(2):
            value = int(matrix[i, j])
            ax.text(
                j,
                i,
                f"{value}\n{value / matrix[i].sum():.1%} da linha",
                ha="center",
                va="center",
                color="white" if value > matrix.max() * 0.48 else PALETTE["navy"],
                fontsize=20,
                weight="bold",
            )
    ax.text(
        0.5,
        -0.18,
        f"Threshold = {m['threshold']:.4f}  ·  n = {m['n']}",
        transform=ax.transAxes,
        ha="center",
        fontsize=13,
        color=PALETTE["gray"],
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Número de imagens")
    fig.tight_layout()
    save(fig, "02_matriz_confusao.png")


def grouped_roc_band(
    rows: list[dict], replicates: int = 2000, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.array([r["label"] for r in rows], dtype=int)
    scores = np.array([r["probability"] for r in rows], dtype=float)
    groups = np.array([r["bootstrap_group_id"] for r in rows])
    units = np.unique(groups)
    members = {unit: np.flatnonzero(groups == unit) for unit in units}
    grid = np.linspace(0, 1, 201)
    curves = []
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        sampled = rng.choice(units, size=len(units), replace=True)
        idx = np.concatenate([members[unit] for unit in sampled])
        if np.unique(labels[idx]).size < 2:
            continue
        fpr, tpr, _ = roc_curve(labels[idx], scores[idx])
        curves.append(np.interp(grid, fpr, tpr))
    values = np.asarray(curves)
    return grid, np.quantile(values, 0.025, axis=0), np.quantile(values, 0.975, axis=0)


def plot_roc(test: dict, rows: list[dict]) -> None:
    labels = np.array([r["label"] for r in rows], dtype=int)
    scores = np.array([r["probability"] for r in rows], dtype=float)
    fpr, tpr, _ = roc_curve(labels, scores)
    grid, lower, upper = grouped_roc_band(rows)
    ci = test["confidence_intervals"]["roc_auc"]
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.fill_between(
        grid,
        lower,
        upper,
        color=PALETTE["blue"],
        alpha=0.18,
        label="IC 95% da curva · bootstrap por grupo",
    )
    ax.plot(
        fpr,
        tpr,
        color=PALETTE["blue"],
        linewidth=3,
        label=f"ROC-AUC {test['metrics']['roc_auc']:.4f} (IC {ci['lower']:.4f}–{ci['upper']:.4f})",
    )
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color=PALETTE["gray"],
        linewidth=1.5,
        label="Referência aleatória",
    )
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1.01),
        xlabel="Taxa de falsos positivos",
        ylabel="Sensibilidade",
        title="Curva ROC · holdout agrupado",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right", frameon=False, fontsize=11)
    fig.tight_layout()
    save(fig, "03_curva_roc_ic_bootstrap.png")


def distinct_ranked(rows: list[dict], sort_key, count: int) -> list[dict]:
    chosen = []
    used = set()
    for row in sorted(rows, key=sort_key):
        if row["bootstrap_group_id"] in used:
            continue
        chosen.append(row)
        used.add(row["bootstrap_group_id"])
        if len(chosen) == count:
            break
    return chosen


def plot_gradcam(test: dict, rows: list[dict]) -> None:
    threshold = float(test["metrics"]["threshold"])
    for row in rows:
        row["pred"] = int(row["probability"] >= threshold)
    tp = [r for r in rows if r["label"] == 1 and r["pred"] == 1]
    tn = [r for r in rows if r["label"] == 0 and r["pred"] == 0]
    fn = [r for r in rows if r["label"] == 1 and r["pred"] == 0]
    fp = [r for r in rows if r["label"] == 0 and r["pred"] == 1]
    chosen_tp = [
        distinct_ranked(tp, lambda r: abs(r["probability"] - threshold), 1)[0],
        distinct_ranked(tp, lambda r: -r["probability"], len(tp))[0],
    ]
    if chosen_tp[0]["bootstrap_group_id"] == chosen_tp[1]["bootstrap_group_id"]:
        chosen_tp = distinct_ranked(tp, lambda r: -r["probability"], 2)
    chosen = [*chosen_tp, distinct_ranked(tn, lambda r: -r["probability"], 1)[0]]
    chosen += [
        distinct_ranked(fn, lambda r: abs(r["probability"] - threshold), 1)[0],
        distinct_ranked(fn, lambda r: r["probability"], len(fn))[0],
        distinct_ranked(fp, lambda r: -r["probability"], 1)[0],
    ]

    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    train_config = checkpoint["config"]
    train_config["model"]["pretrained"] = False
    model = build_model(train_config).eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    data_config = load_yaml(DATA_CONFIG)
    transform = build_image_transform(data_config, training=False)
    size = int(data_config["image"]["size"])
    root = ROOT / data_config["paths"]["data_root"]
    overlays = []
    for row in chosen:
        image_path = root / row["relative_path"]
        with Image.open(image_path) as image_file:
            resized = image_file.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        original = np.asarray(resized, dtype=np.float32) / 255.0
        image_tensor = transform(resized).requires_grad_(True)
        overlay = create_gradcam(
            model, "mobilenet_v3_small", image_tensor, original, positive_class_index=1
        )
        overlays.append(overlay)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    section_names = ["ACERTOS ILUSTRATIVOS", "ERROS ILUSTRATIVOS"]
    for col, row in enumerate(chosen[:3]):
        pred_name = "PNEUMONIA" if row["pred"] else "NORMAL"
        true_name = "PNEUMONIA" if row["label"] else "NORMAL"
        axes[0, col].imshow(overlays[col])
        axes[0, col].set_title(
            f"Real: {true_name}\nPredito: {pred_name} · p={row['probability']:.3f}",
            fontsize=12,
            color=PALETTE["navy"],
        )
        axes[0, col].axis("off")
    for col, row in enumerate(chosen[3:]):
        pred_name = "PNEUMONIA" if row["pred"] else "NORMAL"
        true_name = "PNEUMONIA" if row["label"] else "NORMAL"
        axes[1, col].imshow(overlays[3 + col])
        axes[1, col].set_title(
            f"Real: {true_name}\nPredito: {pred_name} · p={row['probability']:.3f}",
            fontsize=12,
            color=PALETTE["navy"],
        )
        axes[1, col].axis("off")
    for row_i, label in enumerate(section_names):
        axes[row_i, 0].text(
            -0.12,
            0.5,
            label,
            rotation=90,
            transform=axes[row_i, 0].transAxes,
            ha="center",
            va="center",
            fontsize=12,
            weight="bold",
            color=PALETTE["teal"] if row_i == 0 else PALETTE["coral"],
        )
    fig.suptitle(
        "Grad-CAM exploratório · classe pneumonia",
        fontsize=20,
        color=PALETTE["navy"],
        weight="bold",
    )
    fig.text(
        0.5,
        0.015,
        "Seis imagens do holdout, escolhidas por regra de score; visualização não prova "
        "causalidade nem substitui revisão clínica.",
        ha="center",
        fontsize=11,
        color=PALETTE["gray"],
    )
    fig.tight_layout(rect=[0.02, 0.05, 1, 0.95])
    save(fig, "04_gradcam_6_exemplos.png")


def plot_threshold(val: dict, test: dict) -> None:
    labels = ["Validação", "Holdout final"]
    sens = [100 * val["metrics"]["sensitivity"], 100 * test["metrics"]["sensitivity"]]
    spec = [100 * val["metrics"]["specificity"], 100 * test["metrics"]["specificity"]]
    x = np.arange(2)
    width = 0.34
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.bar(x - width / 2, sens, width, color=PALETTE["teal"], label="Sensibilidade")
    ax.bar(x + width / 2, spec, width, color=PALETTE["gold"], label="Especificidade")
    ax.axhline(
        95, color=PALETTE["coral"], linestyle="--", label="Alvo de sensibilidade na validação: 95%"
    )
    for i, (a, b) in enumerate(zip(sens, spec, strict=True)):
        ax.text(i - width / 2, a + 1, f"{a:.2f}%", ha="center", fontsize=12, weight="bold")
        ax.text(i + width / 2, b + 1, f"{b:.2f}%", ha="center", fontsize=12, weight="bold")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 107)
    ax.set_ylabel("Percentual")
    ax.set_title(
        f"Threshold congelado na validação: {test['metrics']['threshold']:.4f}",
        color=PALETTE["navy"],
        weight="bold",
    )
    ax.legend(frameon=False, loc="lower left")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    save(fig, "05_tradeoff_threshold.png")


def plot_split_change() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle(
        "A decisão mudou a avaliação, não o desempenho do modelo",
        fontsize=20,
        color=PALETTE["navy"],
        weight="bold",
    )
    before = [("Treino original", 5232, PALETTE["blue"]), ("Teste original", 624, PALETTE["gold"])]
    after = [
        ("Treino agrupado", 4119, PALETTE["blue"]),
        ("Validação", 867, PALETTE["teal"]),
        ("Holdout", 870, PALETTE["gold"]),
    ]
    for ax, items, title in [
        (axes[0], before, "Antes · pastas originais"),
        (axes[1], after, "Depois · split agrupado"),
    ]:
        names = [item[0] for item in items]
        vals = [item[1] for item in items]
        colors = [item[2] for item in items]
        bars = ax.barh(names, vals, color=colors, height=0.58)
        ax.invert_yaxis()
        ax.set_xlim(0, 6100)
        ax.set_title(title, color=PALETTE["navy"], weight="bold")
        ax.set_xlabel("Imagens")
        ax.grid(axis="x", alpha=0.2)
        for bar, val in zip(bars, vals, strict=True):
            ax.text(
                val + 80,
                bar.get_y() + bar.get_height() / 2,
                f"{val:,}".replace(",", "."),
                va="center",
                weight="bold",
                color=PALETTE["navy"],
            )
    overlap = 170
    axes[0].text(
        0.02,
        -0.24,
        f"{overlap} tokens de grupo inferidos em comum entre train/test; "
        "nenhum ID clínico foi verificado.",
        transform=axes[0].transAxes,
        color=PALETTE["coral"],
        fontsize=11,
    )
    axes[1].text(
        0.02,
        -0.24,
        "Zero grupos compartilhados nos novos splits; grupos são proxies de filename/hash.",
        transform=axes[1].transAxes,
        color=PALETTE["gray"],
        fontsize=11,
    )
    fig.text(
        0.5,
        0.02,
        "Este controle reduz leakage detectável por proxies; não demonstra independência "
        "clínica nem mede ganho de acurácia.",
        ha="center",
        fontsize=11,
        color=PALETTE["gray"],
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.92])
    save(fig, "06_antes_depois_split_agrupado.png")


def plot_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    fig.suptitle(
        "Do raio-X à demo: pipeline do projeto",
        fontsize=23,
        color=PALETTE["navy"],
        weight="bold",
        y=0.96,
    )
    boxes = [
        (
            0.5,
            5.2,
            2.6,
            1.7,
            "DADOS",
            "5.856 imagens\nKermany / Mendeley\nNORMAL × PNEUMONIA",
            PALETTE["blue"],
        ),
        (
            3.7,
            5.2,
            2.8,
            1.7,
            "AUDITORIA",
            "Integridade + hashes\n30 grupos de duplicatas\n170 tokens com overlap",
            PALETTE["teal"],
        ),
        (
            7.1,
            5.2,
            2.8,
            1.7,
            "SPLIT AGRUPADO",
            "4.119 treino\n867 validação\n870 holdout",
            PALETTE["gold"],
        ),
        (
            10.5,
            5.2,
            2.4,
            1.7,
            "MODELO",
            "MobileNetV3 Small\nPesos ImageNet\nbackbone congelado",
            PALETTE["blue"],
        ),
        (
            13.5,
            5.2,
            2.0,
            1.7,
            "DEMO",
            "Streamlit\nscore + Grad-CAM\nOOD exploratório",
            PALETTE["teal"],
        ),
    ]
    for x, y, w, h, title, body, color in boxes:
        rect = plt.Rectangle(
            (x, y),
            w,
            h,
            facecolor=PALETTE["pale"],
            edgecolor=color,
            linewidth=2.4,
            joinstyle="round",
        )
        ax.add_patch(rect)
        ax.text(x + 0.16, y + h - 0.32, title, color=color, weight="bold", fontsize=13)
        ax.text(
            x + 0.16,
            y + 0.3,
            body,
            color=PALETTE["navy"],
            fontsize=11.4,
            va="bottom",
            linespacing=1.35,
        )
    for x1, x2 in [(3.1, 3.7), (6.5, 7.1), (9.9, 10.5), (12.9, 13.5)]:
        ax.annotate(
            "",
            xy=(x2 - 0.06, 6.05),
            xytext=(x1 + 0.06, 6.05),
            arrowprops={"arrowstyle": "->", "lw": 2, "color": PALETTE["gray"]},
        )
    bottom = [
        (
            1.0,
            "TREINO",
            "13 épocas concluídas\ncheckpoint selecionado: época 11\n"
            "execução interrompida manualmente",
        ),
        (5.1, "CALIBRAÇÃO", "Temperature scaling\nthreshold = 0,2522\nselecionados na validação"),
        (
            9.2,
            "AVALIAÇÃO FINAL",
            "Holdout usado uma vez\nIC por bootstrap agrupado\nsem validação externa",
        ),
        (13.0, "RESULTADO", "Sensibilidade 95,85%\nEspecificidade 92,27%\nuso somente educacional"),
    ]
    for x, title, body in bottom:
        ax.text(x, 3.7, title, fontsize=13, color=PALETTE["navy"], weight="bold")
        ax.text(x, 2.75, body, fontsize=11.2, color=PALETTE["gray"], linespacing=1.4)
    for x1, x2 in [(3.0, 5.0), (7.5, 9.1), (11.2, 12.9)]:
        ax.annotate(
            "",
            xy=(x2, 3.45),
            xytext=(x1, 3.45),
            arrowprops={"arrowstyle": "->", "lw": 1.8, "color": PALETTE["gray"]},
        )
    ax.text(
        0.7,
        0.75,
        "Limite essencial: um dataset retrospectivo de um centro; grupos aproximados por "
        "filenames/hashes; sem uso clínico.",
        fontsize=13,
        color=PALETTE["coral"],
        weight="bold",
    )
    save(fig, "07_pipeline_uma_pagina.png")


def plot_learning_curve() -> None:
    metrics_path = ROOT / "runs" / RUN_NAME / "metrics.jsonl"
    records = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    epochs = [r["epoch"] for r in records]
    train_auc = [r["train"]["roc_auc"] for r in records]
    val_auc = [r["validation"]["roc_auc"] for r in records]
    best_epoch = int(np.argmax(val_auc)) + 1
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.plot(epochs, train_auc, marker="o", color=PALETTE["blue"], label="Treino")
    ax.plot(epochs, val_auc, marker="o", color=PALETTE["teal"], label="Validação")
    ax.axvline(
        best_epoch,
        linestyle="--",
        color=PALETTE["gold"],
        label=f"Melhor validação: época {best_epoch}",
    )
    ax.set(
        xticks=epochs,
        xlabel="Época",
        ylabel="ROC-AUC registrada",
        ylim=(0.88, 1.0),
        title="Curva de aprendizado · 13 épocas concluídas",
    )
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    ax.text(
        0.02,
        0.04,
        "Treino interrompido manualmente após a época 13; early stopping não foi concluído.",
        transform=ax.transAxes,
        color=PALETTE["coral"],
        fontsize=11,
    )
    fig.tight_layout()
    save(fig, "08_curva_de_aprendizado.png")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()
    test = read_json(EVAL_DIR / "test_results.json")
    val = read_json(EVAL_DIR / "val_results.json")
    rows = read_predictions(EVAL_DIR / "test_predictions.csv")
    plot_model_comparison(test)
    plot_confusion(test)
    plot_roc(test, rows)
    plot_gradcam(test, rows)
    plot_threshold(val, test)
    plot_split_change()
    plot_pipeline()
    plot_learning_curve()
    print(f"Saved portfolio figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
