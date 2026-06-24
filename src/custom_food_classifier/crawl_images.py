from __future__ import annotations

import argparse
import shutil
from math import ceil
from pathlib import Path

from icrawler.builtin import BingImageCrawler

from custom_food_classifier.categories import DEFAULT_CATEGORIES
from custom_food_classifier.utils import (
    clean_and_deduplicate_images,
    ensure_dir,
    iter_image_files,
    write_json,
)


def get_next_image_index(category_dir: Path) -> int:
    max_index = 0
    for file_path in iter_image_files(category_dir):
        try:
            max_index = max(max_index, int(file_path.stem))
        except ValueError:
            continue
    return max_index + 1


def merge_keyword_downloads(temp_dir: Path, category_dir: Path) -> int:
    moved_count = 0
    next_index = get_next_image_index(category_dir)

    for file_path in iter_image_files(temp_dir):
        new_name = f"{next_index:06d}{file_path.suffix.lower()}"
        shutil.move(str(file_path), str(category_dir / new_name))
        next_index += 1
        moved_count += 1

    shutil.rmtree(temp_dir, ignore_errors=True)
    return moved_count


def crawl_single_category(
    slug: str,
    display_name: str,
    keywords: list[str],
    output_root: str,
    target_count: int,
    oversample_factor: float,
    clear_existing: bool,
) -> dict[str, object]:
    category_dir = ensure_dir(Path(output_root) / slug)
    if clear_existing and category_dir.exists():
        shutil.rmtree(category_dir)
        category_dir.mkdir(parents=True, exist_ok=True)

    current_count = len(list(iter_image_files(category_dir)))
    remaining_target = max(0, target_count - current_count)

    if remaining_target > 0:
        download_target = ceil(remaining_target * oversample_factor)
        keywords_left = len(keywords)

        for index, keyword in enumerate(keywords, start=1):
            current_count = len(list(iter_image_files(category_dir)))
            remaining_target = max(0, target_count - current_count)
            if remaining_target <= 0:
                break

            keywords_remaining = max(1, keywords_left - index + 1)
            per_keyword = ceil((remaining_target * oversample_factor) / keywords_remaining)
            temp_dir = category_dir / f"_tmp_keyword_{index:02d}"
            ensure_dir(temp_dir)

            crawler = BingImageCrawler(
                storage={"root_dir": str(temp_dir)},
                downloader_threads=4,
                parser_threads=2,
            )
            crawler.crawl(keyword=keyword, max_num=per_keyword)
            merge_keyword_downloads(temp_dir, category_dir)

    summary = clean_and_deduplicate_images(category_dir)
    summary.update(
        {
            "slug": slug,
            "display_name": display_name,
            "target_count": target_count,
            "current_count": summary["valid_images"],
            "keywords": keywords,
        }
    )

    if summary["valid_images"] > target_count:
        files = list(iter_image_files(category_dir))
        extras = files[target_count:]
        for file_path in extras:
            file_path.unlink(missing_ok=True)
        summary["valid_images"] = target_count
        summary["current_count"] = target_count

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="爬取五类菜品图像")
    parser.add_argument("--output-root", default="./scraped_food_data/raw")
    parser.add_argument("--images-per-class", type=int, default=500)
    parser.add_argument("--oversample-factor", type=float, default=1.4)
    parser.add_argument("--clear-existing", action="store_true")
    args = parser.parse_args()

    output_root = ensure_dir(args.output_root)
    all_results = []

    for category in DEFAULT_CATEGORIES:
        result = crawl_single_category(
            slug=category["slug"],
            display_name=category["display_name"],
            keywords=category["keywords"],
            output_root=str(output_root),
            target_count=args.images_per_class,
            oversample_factor=args.oversample_factor,
            clear_existing=args.clear_existing,
        )
        all_results.append(result)
        print(
            f"{result['display_name']}: "
            f"{result['current_count']} / {result['target_count']} 张"
        )

    write_json(
        {"categories": all_results},
        Path(output_root) / "crawl_summary.json",
    )


if __name__ == "__main__":
    main()
