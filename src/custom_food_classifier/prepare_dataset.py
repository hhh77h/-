from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from custom_food_classifier.categories import DEFAULT_CATEGORIES
from custom_food_classifier.utils import ensure_dir, iter_image_files, write_json


def split_counts(total: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count
    return train_count, val_count, test_count


def copy_split(
    images: list[Path],
    output_root: str | Path,
    class_name: str,
    train_ratio: float,
    val_ratio: float,
) -> dict[str, int]:
    train_count, val_count, _ = split_counts(len(images), train_ratio, val_ratio)
    subsets = {
        "train": images[:train_count],
        "val": images[train_count : train_count + val_count],
        "test": images[train_count + val_count :],
    }

    summary = {}
    for subset, subset_files in subsets.items():
        subset_dir = ensure_dir(Path(output_root) / subset / class_name)
        for file_path in subset_files:
            shutil.copy2(file_path, subset_dir / file_path.name)
        summary[subset] = len(subset_files)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="将原始爬取图片划分为训练集/验证集/测试集")
    parser.add_argument("--raw-root", default="./scraped_food_data/raw")
    parser.add_argument("--output-root", default="./scraped_food_data/split")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clear-existing", action="store_true")
    args = parser.parse_args()

    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio 必须小于 1.0")

    output_root = Path(args.output_root)
    if args.clear_existing and output_root.exists():
        shutil.rmtree(output_root)

    random.seed(args.seed)
    summary = {}

    for category in DEFAULT_CATEGORIES:
        class_dir = Path(args.raw_root) / category["slug"]
        images = list(iter_image_files(class_dir))
        random.shuffle(images)

        if not images:
            raise FileNotFoundError(f"类别目录为空: {class_dir}")

        summary[category["slug"]] = copy_split(
            images=images,
            output_root=output_root,
            class_name=category["slug"],
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )
        print(f"{category['display_name']}: {summary[category['slug']]}")

    write_json(summary, Path(args.output_root) / "split_summary.json")


if __name__ == "__main__":
    main()
