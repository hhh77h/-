from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from torch import nn
from tqdm.auto import tqdm

from custom_food_classifier.dataset import build_dataloaders
from custom_food_classifier.model import create_resnet18_classifier
from custom_food_classifier.utils import ensure_dir, write_json


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_one_epoch(model, dataloader, criterion, optimizer, device):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress = tqdm(dataloader, leave=False)
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            outputs = model(images)
            loss = criterion(outputs, labels)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        predictions = outputs.argmax(dim=1)
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (predictions == labels).sum().item()
        total_samples += batch_size
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, dataloader, class_names, device):
    model.eval()
    all_labels = []
    all_predictions = []

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        predictions = outputs.argmax(dim=1).cpu()
        all_predictions.extend(predictions.tolist())
        all_labels.extend(labels.tolist())

    return classification_report(
        all_labels,
        all_predictions,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )


def plot_history(history: dict[str, list[float]], save_path: str) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="基于自建五类菜品数据集训练 ResNet18")
    parser.add_argument("--dataset-root", default="./scraped_food_data/split")
    parser.add_argument("--output-root", default="./scraped_food_data/outputs")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--step-size", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    output_root = ensure_dir(args.output_root)
    device = resolve_device(args.device)

    (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
    ) = build_dataloaders(
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )

    class_names = train_dataset.classes
    model = create_resnet18_classifier(
        num_classes=len(class_names),
        pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.step_size,
        gamma=args.gamma,
    )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    checkpoint_path = Path(output_root) / "best_model.pth"

    for epoch in range(args.epochs):
        print(f"Epoch [{epoch + 1}/{args.epochs}]")
        train_loss, train_acc = run_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = run_one_epoch(model, val_loader, criterion, None, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "image_size": args.image_size,
                },
                checkpoint_path,
            )

    model_bundle = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(model_bundle["model_state_dict"])
    report = evaluate(model, test_loader, class_names, device)

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(Path(output_root) / "classification_report.csv", encoding="utf-8-sig")
    Path(output_root, "history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )
    plot_history(history, str(Path(output_root) / "training_curves.png"))
    write_json(
        {
            "num_classes": len(class_names),
            "class_names": class_names,
            "num_train_samples": len(train_dataset),
            "num_val_samples": len(val_dataset),
            "num_test_samples": len(test_dataset),
            "best_val_acc": best_val_acc,
            "checkpoint_path": str(checkpoint_path),
        },
        Path(output_root) / "summary.json",
    )


if __name__ == "__main__":
    main()
