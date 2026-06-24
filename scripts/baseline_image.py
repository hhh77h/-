from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from baseline_model import load_split_dataset
from custom_food_classifier.categories import DEFAULT_CATEGORIES

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(r"e:\深度学习课设")
RAW_ROOT = PROJECT_ROOT / "scraped_food_data" / "raw"
SPLIT_ROOT = PROJECT_ROOT / "scraped_food_data" / "split"
BASELINE_ROOT = PROJECT_ROOT / "baseline_model"

SUMMARY_PATH = BASELINE_ROOT / "summary.json"
HISTORY_PATH = BASELINE_ROOT / "history.json"
REPORT_PATH = BASELINE_ROOT / "classification_report.csv"
MODEL_PATH = BASELINE_ROOT / "best_model.joblib"

VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_label_mapping() -> dict[str, str]:
    return {item["slug"]: item["display_name"] for item in DEFAULT_CATEGORIES}


def _image_files(directory: Path) -> list[Path]:
    return sorted([p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in VALID_SUFFIXES])


def _finalize_figure(save_path: str | Path | None, show: bool) -> None:
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def collect_class_counts(data_root: str | Path) -> dict[str, int]:
    data_root = Path(data_root)
    label_mapping = get_label_mapping()
    counts: dict[str, int] = {}
    for class_dir in sorted([p for p in data_root.iterdir() if p.is_dir()]):
        counts[label_mapping.get(class_dir.name, class_dir.name)] = len(_image_files(class_dir))
    return counts


def collect_split_counts(split_root: str | Path = SPLIT_ROOT) -> dict[str, dict[str, int]]:
    split_root = Path(split_root)
    label_mapping = get_label_mapping()
    split_counts: dict[str, dict[str, int]] = {}
    for subset in ["train", "val", "test"]:
        subset_dir = split_root / subset
        subset_counts = {}
        if subset_dir.exists():
            for class_dir in sorted([p for p in subset_dir.iterdir() if p.is_dir()]):
                subset_counts[label_mapping.get(class_dir.name, class_dir.name)] = len(_image_files(class_dir))
        split_counts[subset] = subset_counts
    return split_counts


def load_baseline_summary(summary_path: str | Path = SUMMARY_PATH) -> dict:
    return pd.read_json(Path(summary_path), typ="series").to_dict()


def load_baseline_history(history_path: str | Path = HISTORY_PATH) -> dict:
    return pd.read_json(Path(history_path), typ="series").to_dict()


def load_baseline_report(report_csv_path: str | Path = REPORT_PATH) -> pd.DataFrame:
    return pd.read_csv(report_csv_path, index_col=0)


def load_baseline_model(model_path: str | Path = MODEL_PATH):
    return joblib.load(model_path)


def get_baseline_dataset(
    split_root: str | Path = SPLIT_ROOT,
    baseline_root: str | Path = BASELINE_ROOT,
) -> dict:
    summary = load_baseline_summary(Path(baseline_root) / "summary.json")
    return load_split_dataset(
        split_root=split_root,
        image_size=int(summary.get("image_size", 64)),
        color_mode=summary.get("color_mode", "grayscale"),
    )


def plot_raw_dataset_distribution(
    raw_root: str | Path = RAW_ROOT,
    save_path: str | Path | None = None,
    show: bool = True,
) -> dict[str, int]:
    counts = collect_class_counts(raw_root)
    plt.figure(figsize=(10, 6))
    plt.bar(counts.keys(), counts.values(), color="#4C72B0")
    plt.title("原始数据集类别分布")
    plt.xlabel("类别")
    plt.ylabel("图片数量")
    plt.xticks(rotation=20)
    for i, value in enumerate(counts.values()):
        plt.text(i, value + 1, str(value), ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    _finalize_figure(save_path, show)
    return counts


def plot_split_distribution(
    split_root: str | Path = SPLIT_ROOT,
    save_path: str | Path | None = None,
    show: bool = True,
) -> dict[str, dict[str, int]]:
    split_counts = collect_split_counts(split_root)
    class_names = list(split_counts["train"].keys())
    x = np.arange(len(class_names))
    width = 0.24

    train_values = [split_counts["train"].get(name, 0) for name in class_names]
    val_values = [split_counts["val"].get(name, 0) for name in class_names]
    test_values = [split_counts["test"].get(name, 0) for name in class_names]

    plt.figure(figsize=(11, 6))
    plt.bar(x - width, train_values, width=width, label="Train", color="#4C72B0")
    plt.bar(x, val_values, width=width, label="Val", color="#55A868")
    plt.bar(x + width, test_values, width=width, label="Test", color="#C44E52")
    plt.xticks(x, class_names, rotation=20)
    plt.ylabel("图片数量")
    plt.xlabel("类别")
    plt.title("Baseline 数据集划分分布")
    plt.legend()
    plt.tight_layout()
    _finalize_figure(save_path, show)
    return split_counts


def plot_sample_images(
    data_root: str | Path = SPLIT_ROOT / "train",
    samples_per_class: int = 4,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    data_root = Path(data_root)
    label_mapping = get_label_mapping()
    class_dirs = sorted([p for p in data_root.iterdir() if p.is_dir()])
    rows = len(class_dirs)
    cols = samples_per_class

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    if rows == 1:
        axes = np.array([axes])

    for row, class_dir in enumerate(class_dirs):
        image_paths = _image_files(class_dir)[:samples_per_class]
        display_name = label_mapping.get(class_dir.name, class_dir.name)

        for col in range(cols):
            ax = axes[row, col]
            if col < len(image_paths):
                image = Image.open(image_paths[col]).convert("RGB")
                ax.imshow(image)
                ax.set_title(f"{display_name} - 样本{col + 1}")
            ax.axis("off")

    plt.suptitle("Baseline 样本可视化", fontsize=16)
    plt.tight_layout()
    _finalize_figure(save_path, show)


def plot_baseline_loss_curve(
    history_path: str | Path = HISTORY_PATH,
    save_path: str | Path | None = None,
    show: bool = True,
) -> dict:
    history = load_baseline_history(history_path)
    loss_curve = history.get("loss_curve", [])
    if not loss_curve:
        raise ValueError("history.json 中未找到 loss_curve。")

    iterations = np.arange(1, len(loss_curve) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(iterations, loss_curve, marker="o", color="#4C72B0")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Baseline MLP 训练损失曲线")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    _finalize_figure(save_path, show)
    return history


def plot_baseline_classification_report_bars(
    report_csv_path: str | Path = REPORT_PATH,
    save_path: str | Path | None = None,
    show: bool = True,
) -> pd.DataFrame:
    df = load_baseline_report(report_csv_path)
    label_mapping = get_label_mapping()

    class_df = df.loc[[idx for idx in df.index if idx in label_mapping]].copy()
    class_df["中文类别"] = [label_mapping[idx] for idx in class_df.index]

    x = np.arange(len(class_df))
    width = 0.25

    plt.figure(figsize=(12, 6))
    plt.bar(x - width, class_df["precision"], width=width, label="Precision", color="#4C72B0")
    plt.bar(x, class_df["recall"], width=width, label="Recall", color="#55A868")
    plt.bar(x + width, class_df["f1-score"], width=width, label="F1-score", color="#C44E52")
    plt.xticks(x, class_df["中文类别"], rotation=20)
    plt.ylim(0, 1.05)
    plt.ylabel("指标值")
    plt.title("Baseline 各类别 Precision / Recall / F1-score")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _finalize_figure(save_path, show)
    return class_df


def get_baseline_predictions(
    split_root: str | Path = SPLIT_ROOT,
    baseline_root: str | Path = BASELINE_ROOT,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    dataset = get_baseline_dataset(split_root, baseline_root)
    model = load_baseline_model(Path(baseline_root) / "best_model.joblib")

    y_true = dataset["y_test"]
    y_pred = model.predict(dataset["x_test"])
    class_names = dataset["class_names"]
    test_paths = dataset["test_paths"]
    return y_true, y_pred, class_names, test_paths


def plot_baseline_confusion_matrix(
    split_root: str | Path = SPLIT_ROOT,
    baseline_root: str | Path = BASELINE_ROOT,
    normalize: str | None = "true",
    save_path: str | Path | None = None,
    show: bool = True,
) -> np.ndarray:
    y_true, y_pred, class_names, _ = get_baseline_predictions(split_root, baseline_root)
    label_mapping = get_label_mapping()
    display_labels = [label_mapping.get(name, name) for name in class_names]

    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(cmap="Blues", values_format=".2f" if normalize else "d", ax=ax, colorbar=True)
    plt.title("Baseline MLP 测试集混淆矩阵")
    plt.xticks(rotation=20)
    plt.tight_layout()
    _finalize_figure(save_path, show)
    return cm


def plot_baseline_misclassified_examples(
    split_root: str | Path = SPLIT_ROOT,
    baseline_root: str | Path = BASELINE_ROOT,
    max_images: int = 9,
    save_path: str | Path | None = None,
    show: bool = True,
) -> list[dict]:
    y_true, y_pred, class_names, test_paths = get_baseline_predictions(split_root, baseline_root)
    label_mapping = get_label_mapping()

    misclassified = []
    for image_path, true_idx, pred_idx in zip(test_paths, y_true, y_pred):
        if int(true_idx) != int(pred_idx):
            misclassified.append(
                {
                    "image_path": image_path,
                    "true_label": label_mapping.get(class_names[int(true_idx)], class_names[int(true_idx)]),
                    "pred_label": label_mapping.get(class_names[int(pred_idx)], class_names[int(pred_idx)]),
                }
            )
        if len(misclassified) >= max_images:
            break

    cols = min(3, max_images)
    rows = int(np.ceil(max(1, len(misclassified)) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.8 * rows))
    axes = np.array(axes).reshape(rows, cols)

    for ax in axes.ravel():
        ax.axis("off")

    for ax, item in zip(axes.ravel(), misclassified):
        image = Image.open(item["image_path"]).convert("RGB")
        ax.imshow(image)
        ax.set_title(f"真实: {item['true_label']}\n预测: {item['pred_label']}", fontsize=10)
        ax.axis("off")

    plt.suptitle("Baseline MLP 误分类样本", fontsize=16)
    plt.tight_layout()
    _finalize_figure(save_path, show)
    return misclassified


def plot_baseline_summary_bar(
    summary_path: str | Path = SUMMARY_PATH,
    save_path: str | Path | None = None,
    show: bool = True,
) -> dict:
    summary = load_baseline_summary(summary_path)
    names = ["Train Acc", "Val Acc", "Test Acc"]
    values = [
        float(summary.get("train_accuracy", 0.0)),
        float(summary.get("val_accuracy", 0.0)),
        float(summary.get("test_accuracy", 0.0)),
    ]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, values, color=["#4C72B0", "#55A868", "#C44E52"])
    plt.ylim(0, 1.05)
    plt.ylabel("Accuracy")
    plt.title("Baseline MLP 数据集准确率对比")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.4f}", ha="center")
    plt.tight_layout()
    _finalize_figure(save_path, show)
    return summary


def generate_all_baseline_figures(
    figure_dir: str | Path = PROJECT_ROOT / "baseline_report_figures",
    sample_data_root: str | Path = SPLIT_ROOT / "train",
) -> dict[str, str]:
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    plot_raw_dataset_distribution(save_path=figure_dir / "01_raw_dataset_distribution.png", show=False)
    plot_split_distribution(save_path=figure_dir / "02_split_distribution.png", show=False)
    plot_sample_images(data_root=sample_data_root, save_path=figure_dir / "03_sample_images.png", show=False)
    plot_baseline_loss_curve(save_path=figure_dir / "04_baseline_loss_curve.png", show=False)
    plot_baseline_summary_bar(save_path=figure_dir / "05_baseline_accuracy_bar.png", show=False)
    plot_baseline_classification_report_bars(save_path=figure_dir / "06_baseline_class_metrics.png", show=False)
    plot_baseline_confusion_matrix(save_path=figure_dir / "07_baseline_confusion_matrix.png", show=False)
    plot_baseline_misclassified_examples(save_path=figure_dir / "08_baseline_misclassified_examples.png", show=False)

    return {
        "raw_distribution": str(figure_dir / "01_raw_dataset_distribution.png"),
        "split_distribution": str(figure_dir / "02_split_distribution.png"),
        "sample_images": str(figure_dir / "03_sample_images.png"),
        "loss_curve": str(figure_dir / "04_baseline_loss_curve.png"),
        "accuracy_bar": str(figure_dir / "05_baseline_accuracy_bar.png"),
        "class_metrics": str(figure_dir / "06_baseline_class_metrics.png"),
        "confusion_matrix": str(figure_dir / "07_baseline_confusion_matrix.png"),
        "misclassified_examples": str(figure_dir / "08_baseline_misclassified_examples.png"),
    }


if __name__ == "__main__":
    paths = generate_all_baseline_figures()
    print("已生成 Baseline 图表：")
    for name, path in paths.items():
        print(f"{name}: {path}")