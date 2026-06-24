from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_step(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="一键执行爬取、划分与训练")
    parser.add_argument("--images-per-class", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--raw-root", default="./scraped_food_data/raw")
    parser.add_argument("--split-root", default="./scraped_food_data/split")
    parser.add_argument("--output-root", default="./scraped_food_data/outputs")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    python = sys.executable

    run_step(
        [
            python,
            "-m",
            "custom_food_classifier.crawl_images",
            "--output-root",
            str(project_root / args.raw_root),
            "--images-per-class",
            str(args.images_per_class),
            "--clear-existing",
        ]
    )
    run_step(
        [
            python,
            "-m",
            "custom_food_classifier.prepare_dataset",
            "--raw-root",
            str(project_root / args.raw_root),
            "--output-root",
            str(project_root / args.split_root),
            "--clear-existing",
        ]
    )
    run_step(
        [
            python,
            "-m",
            "custom_food_classifier.train",
            "--dataset-root",
            str(project_root / args.split_root),
            "--output-root",
            str(project_root / args.output_root),
            "--epochs",
            str(args.epochs),
        ]
    )


if __name__ == "__main__":
    main()
