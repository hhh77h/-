from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_image_files(directory: Path) -> list[Path]:
    return sorted([p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in VALID_SUFFIXES])


def load_split_dataset(
    split_root: str | Path,
    image_size: int = 64,
    color_mode: str = "grayscale",
) -> dict[str, Any]:
    split_root = Path(split_root)
    train_dir = split_root / "train"
    class_names = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    def read_subset(subset: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
        subset_dir = split_root / subset
        features: list[np.ndarray] = []
        labels: list[int] = []
        paths: list[str] = []

        for class_name in class_names:
            class_dir = subset_dir / class_name
            if not class_dir.exists():
                continue
            for image_path in list_image_files(class_dir):
                try:
                    image = Image.open(image_path)
                    image = image.convert("L" if color_mode == "grayscale" else "RGB")
                    image = image.resize((image_size, image_size))
                    array = np.asarray(image, dtype=np.float32) / 255.0
                    features.append(array.reshape(-1))
                    labels.append(class_to_idx[class_name])
                    paths.append(str(image_path))
                except Exception:
                    continue

        if not features:
            raise RuntimeError(f"{subset_dir} 中没有可用图像。")

        return np.stack(features), np.array(labels, dtype=np.int64), paths

    x_train, y_train, train_paths = read_subset("train")
    x_val, y_val, val_paths = read_subset("val")
    x_test, y_test, test_paths = read_subset("test")

    return {
        "class_names": class_names,
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
        "train_paths": train_paths,
        "val_paths": val_paths,
        "test_paths": test_paths,
        "image_size": image_size,
        "color_mode": color_mode,
    }


def build_mlp_model(random_state: int) -> Pipeline:
    estimator = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=64,
        learning_rate_init=1e-3,
        max_iter=100,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=random_state,
        verbose=True,
    )
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )


def plot_confusion_matrix_figure(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    save_path: str | Path,
    normalize: str | None = "true",
) -> None:
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", values_format=".2f" if normalize else "d", ax=ax, colorbar=True)
    plt.title("Baseline MLP 混淆矩阵")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_mlp_history(model: Pipeline, output_dir: Path) -> dict[str, Any]:
    estimator: MLPClassifier = model.named_steps["model"]
    history = {
        "loss_curve": [float(x) for x in estimator.loss_curve_],
        "best_validation_score": (
            float(estimator.best_validation_score_)
            if getattr(estimator, "best_validation_score_", None) is not None
            else None
        ),
        "n_iter": int(estimator.n_iter_),
    }

    (output_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(estimator.loss_curve_) + 1), estimator.loss_curve_, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("MLP 训练损失曲线")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=200, bbox_inches="tight")
    plt.close()

    return history


def evaluate_and_save(
    model: Pipeline,
    dataset: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    class_names = dataset["class_names"]

    y_train_pred = model.predict(dataset["x_train"])
    y_val_pred = model.predict(dataset["x_val"])
    y_test_pred = model.predict(dataset["x_test"])

    train_acc = accuracy_score(dataset["y_train"], y_train_pred)
    val_acc = accuracy_score(dataset["y_val"], y_val_pred)
    test_acc = accuracy_score(dataset["y_test"], y_test_pred)

    report = classification_report(
        dataset["y_test"],
        y_test_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(output_dir / "classification_report.csv", encoding="utf-8-sig")

    plot_confusion_matrix_figure(
        dataset["y_test"],
        y_test_pred,
        class_names=class_names,
        save_path=output_dir / "confusion_matrix.png",
    )

    summary = {
        "model_type": "mlp",
        "image_size": dataset["image_size"],
        "color_mode": dataset["color_mode"],
        "num_classes": len(class_names),
        "class_names": class_names,
        "num_train_samples": int(len(dataset["y_train"])),
        "num_val_samples": int(len(dataset["y_val"])),
        "num_test_samples": int(len(dataset["y_test"])),
        "train_accuracy": float(train_acc),
        "val_accuracy": float(val_acc),
        "test_accuracy": float(test_acc),
        "best_model_path": str(output_dir / "best_model.joblib"),
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def train_baseline(args: argparse.Namespace) -> None:
    output_root = ensure_dir(args.output_root)
    dataset = load_split_dataset(
        split_root=args.dataset_root,
        image_size=args.image_size,
        color_mode=args.color_mode,
    )

    model = build_mlp_model(args.seed)
    print("开始训练 baseline 模型: MLP")
    model.fit(dataset["x_train"], dataset["y_train"])

    val_pred = model.predict(dataset["x_val"])
    val_acc = accuracy_score(dataset["y_val"], val_pred)
    print(f"验证集准确率: {val_acc:.4f}")

    joblib.dump(model, output_root / "best_model.joblib")
    save_mlp_history(model, output_root)
    summary = evaluate_and_save(model, dataset, output_root)

    print("训练完成，结果已保存：")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="训练 MLP 基准模型")
    parser.add_argument(
        "--dataset-root",
        default=r"e:\深度学习课设\scraped_food_data\split",
        help="数据集划分目录，默认使用 train/val/test 结构。",
    )
    parser.add_argument(
        "--output-root",
        default=r"e:\深度学习课设\baseline_model",
        help="Baseline 模型输出目录。",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=64,
        help="输入图像缩放尺寸。",
    )
    parser.add_argument(
        "--color-mode",
        choices=["grayscale", "rgb"],
        default="grayscale",
        help="特征输入颜色模式，默认灰度图。",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    train_baseline(args)


if __name__ == "__main__":
    main()