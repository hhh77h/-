from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from custom_food_classifier.categories import DEFAULT_CATEGORIES
from custom_food_classifier.dataset import IMAGENET_MEAN, IMAGENET_STD
from custom_food_classifier.model import create_resnet18_classifier

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(r"e:\深度学习课设")
RAW_ROOT = PROJECT_ROOT / "scraped_food_data" / "raw"
SPLIT_ROOT = PROJECT_ROOT / "scraped_food_data" / "split"
OUTPUT_ROOT = PROJECT_ROOT / "scraped_food_data" / "outputs"
CHECKPOINT_PATH = OUTPUT_ROOT / "best_model.pth"
HISTORY_PATH = OUTPUT_ROOT / "history.json"
SUMMARY_PATH = OUTPUT_ROOT / "summary.json"
REPORT_PATH = OUTPUT_ROOT / "classification_report.csv"

VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_label_mapping() -> dict[str, str]:
    return {item["slug"]: item["display_name"] for item in DEFAULT_CATEGORIES}


def resolve_device(device: str | None = None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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


def load_image_on_white_background(image_path: str | Path) -> Image.Image:
    image = Image.open(image_path)
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    return image.convert("RGB")


def build_eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def load_checkpoint(checkpoint_path: str | Path = CHECKPOINT_PATH, device: str | None = None) -> dict:
    device_obj = resolve_device(device)
    return torch.load(Path(checkpoint_path), map_location=device_obj)


def load_trained_model(
    checkpoint_path: str | Path = CHECKPOINT_PATH,
    device: str | None = None,
) -> tuple[torch.nn.Module, list[str], int, torch.device]:
    device_obj = resolve_device(device)
    bundle = load_checkpoint(checkpoint_path, device=device)
    class_names = bundle["class_names"]
    image_size = bundle["image_size"]

    model = create_resnet18_classifier(num_classes=len(class_names), pretrained=False).to(device_obj)
    model.load_state_dict(bundle["model_state_dict"])
    model.eval()
    return model, class_names, image_size, device_obj


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
    plt.title("训练/验证/测试集划分分布")
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
                image = load_image_on_white_background(image_paths[col])
                ax.imshow(image)
                ax.set_title(f"{display_name} - 样本{col + 1}")
            ax.axis("off")

    plt.suptitle("数据集样本可视化", fontsize=16)
    plt.tight_layout()
    _finalize_figure(save_path, show)


def plot_training_curves(
    history_path: str | Path = HISTORY_PATH,
    save_path: str | Path | None = None,
    show: bool = True,
) -> dict:
    history = pd.read_json(Path(history_path)).to_dict(orient="list")
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], marker="o", label="Train Loss")
    plt.plot(epochs, history["val_loss"], marker="s", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("损失曲线")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], marker="o", label="Train Acc")
    plt.plot(epochs, history["val_acc"], marker="s", label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("准确率曲线")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    _finalize_figure(save_path, show)
    return history


def plot_classification_report_bars(
    report_csv_path: str | Path = REPORT_PATH,
    save_path: str | Path | None = None,
    show: bool = True,
) -> pd.DataFrame:
    df = pd.read_csv(report_csv_path, index_col=0)
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
    plt.title("各类别 Precision / Recall / F1-score 对比")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _finalize_figure(save_path, show)
    return class_df


def plot_correlation_analysis(
    split_root: str | Path = SPLIT_ROOT,
    report_csv_path: str | Path = REPORT_PATH,
    save_path: str | Path | None = None,
    show: bool = True,
) -> pd.DataFrame:
    split_counts = collect_split_counts(split_root)
    df = pd.read_csv(report_csv_path, index_col=0)
    label_mapping = get_label_mapping()

    class_rows = [idx for idx in df.index if idx in label_mapping]
    class_df = df.loc[class_rows, ["precision", "recall", "f1-score", "support"]].copy()
    class_df["train_samples"] = [split_counts["train"].get(label_mapping[idx], 0) for idx in class_rows]
    class_df["val_samples"] = [split_counts["val"].get(label_mapping[idx], 0) for idx in class_rows]
    class_df["test_samples"] = [split_counts["test"].get(label_mapping[idx], 0) for idx in class_rows]
    class_df["total_samples"] = (
        class_df["train_samples"] + class_df["val_samples"] + class_df["test_samples"]
    )

    corr_df = class_df[[
        "train_samples", "val_samples", "test_samples", "total_samples",
        "precision", "recall", "f1-score", "support"
    ]].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr_df.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_df.columns)))
    ax.set_yticks(range(len(corr_df.index)))
    ax.set_xticklabels(corr_df.columns, rotation=35, ha="right")
    ax.set_yticklabels(corr_df.index)
    ax.set_title("样本数量与分类指标相关性分析")

    for i in range(corr_df.shape[0]):
        for j in range(corr_df.shape[1]):
            ax.text(j, i, f"{corr_df.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    _finalize_figure(save_path, show)
    return corr_df


def _build_eval_loader(
    split_root: str | Path = SPLIT_ROOT,
    split: str = "test",
    checkpoint_path: str | Path = CHECKPOINT_PATH,
    batch_size: int = 16,
) -> tuple[datasets.ImageFolder, DataLoader, int]:
    bundle = load_checkpoint(checkpoint_path)
    image_size = bundle["image_size"]
    dataset = datasets.ImageFolder(Path(split_root) / split, transform=build_eval_transform(image_size))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return dataset, loader, image_size


def get_predictions(
    split_root: str | Path = SPLIT_ROOT,
    split: str = "test",
    checkpoint_path: str | Path = CHECKPOINT_PATH,
    device: str | None = None,
    batch_size: int = 16,
) -> tuple[list[int], list[int], list[str]]:
    model, class_names, _, device_obj = load_trained_model(checkpoint_path, device)
    dataset, loader, _ = _build_eval_loader(split_root, split, checkpoint_path, batch_size)

    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device_obj)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.tolist())

    return y_true, y_pred, dataset.classes


def plot_confusion_matrix(
    split_root: str | Path = SPLIT_ROOT,
    split: str = "test",
    checkpoint_path: str | Path = CHECKPOINT_PATH,
    device: str | None = None,
    normalize: str | None = "true",
    save_path: str | Path | None = None,
    show: bool = True,
) -> np.ndarray:
    y_true, y_pred, class_names = get_predictions(split_root, split, checkpoint_path, device)
    label_mapping = get_label_mapping()
    display_labels = [label_mapping.get(name, name) for name in class_names]

    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(cmap="Blues", values_format=".2f" if normalize else "d", ax=ax, colorbar=True)
    plt.title("测试集混淆矩阵")
    plt.xticks(rotation=20)
    plt.tight_layout()
    _finalize_figure(save_path, show)
    return cm


def plot_misclassified_examples(
    split_root: str | Path = SPLIT_ROOT,
    split: str = "test",
    checkpoint_path: str | Path = CHECKPOINT_PATH,
    device: str | None = None,
    max_images: int = 9,
    save_path: str | Path | None = None,
    show: bool = True,
) -> list[dict]:
    model, class_names, image_size, device_obj = load_trained_model(checkpoint_path, device)
    label_mapping = get_label_mapping()
    transform = build_eval_transform(image_size)

    image_folder = datasets.ImageFolder(Path(split_root) / split)
    samples = image_folder.samples
    misclassified: list[dict] = []

    with torch.no_grad():
        for image_path, true_idx in samples:
            image = load_image_on_white_background(image_path)
            tensor = transform(image).unsqueeze(0).to(device_obj)
            pred_idx = model(tensor).argmax(dim=1).item()
            if pred_idx != true_idx:
                misclassified.append(
                    {
                        "image_path": image_path,
                        "true_label": label_mapping.get(class_names[true_idx], class_names[true_idx]),
                        "pred_label": label_mapping.get(class_names[pred_idx], class_names[pred_idx]),
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
        image = load_image_on_white_background(item["image_path"])
        ax.imshow(image)
        ax.set_title(f"真实: {item['true_label']}\n预测: {item['pred_label']}", fontsize=10)
        ax.axis("off")

    plt.suptitle("误分类样本分析", fontsize=16)
    plt.tight_layout()
    _finalize_figure(save_path, show)
    return misclassified


def plot_prediction_for_single_image(
    image_path: str | Path,
    checkpoint_path: str | Path = CHECKPOINT_PATH,
    device: str | None = None,
    top_k: int = 5,
    save_path: str | Path | None = None,
    show: bool = True,
) -> list[tuple[str, float]]:
    model, class_names, image_size, device_obj = load_trained_model(checkpoint_path, device)
    label_mapping = get_label_mapping()
    image = load_image_on_white_background(image_path)
    tensor = build_eval_transform(image_size)(image).unsqueeze(0).to(device_obj)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
        scores, indices = torch.topk(probs, k=min(top_k, len(class_names)))

    results = [
        (label_mapping.get(class_names[idx], class_names[idx]), float(score))
        for score, idx in zip(scores.cpu().tolist(), indices.cpu().tolist())
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(image)
    axes[0].set_title("输入图像")
    axes[0].axis("off")

    labels = [item[0] for item in results]
    values = [item[1] for item in results]
    axes[1].barh(labels[::-1], values[::-1], color="#4C72B0")
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("概率")
    axes[1].set_title("Top-K 预测结果")
    for i, v in enumerate(values[::-1]):
        axes[1].text(v + 0.01, i, f"{v:.3f}", va="center")

    plt.tight_layout()
    _finalize_figure(save_path, show)
    return results


def generate_all_report_figures(
    figure_dir: str | Path = PROJECT_ROOT / "report_figures",
    sample_data_root: str | Path = SPLIT_ROOT / "train",
) -> dict[str, str]:
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    plot_raw_dataset_distribution(save_path=figure_dir / "01_raw_dataset_distribution.png", show=False)
    plot_split_distribution(save_path=figure_dir / "02_split_distribution.png", show=False)
    plot_sample_images(data_root=sample_data_root, save_path=figure_dir / "03_sample_images.png", show=False)
    plot_training_curves(save_path=figure_dir / "04_training_curves.png", show=False)
    plot_classification_report_bars(save_path=figure_dir / "05_class_metrics.png", show=False)
    plot_confusion_matrix(save_path=figure_dir / "06_confusion_matrix.png", show=False)
    plot_misclassified_examples(save_path=figure_dir / "07_misclassified_examples.png", show=False)
    plot_correlation_analysis(save_path=figure_dir / "08_correlation_analysis.png", show=False)

    return {
        "raw_distribution": str(figure_dir / "01_raw_dataset_distribution.png"),
        "split_distribution": str(figure_dir / "02_split_distribution.png"),
        "sample_images": str(figure_dir / "03_sample_images.png"),
        "training_curves": str(figure_dir / "04_training_curves.png"),
        "class_metrics": str(figure_dir / "05_class_metrics.png"),
        "confusion_matrix": str(figure_dir / "06_confusion_matrix.png"),
        "misclassified_examples": str(figure_dir / "07_misclassified_examples.png"),
        "correlation_analysis": str(figure_dir / "08_correlation_analysis.png"),
    }


if __name__ == "__main__":
    paths = generate_all_report_figures()
    print("已生成报告图表：")
    for name, path in paths.items():
        print(f"{name}: {path}")