from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image, UnidentifiedImageError

VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def iter_image_files(directory: str | Path):
    directory = Path(directory)
    for file_path in sorted(directory.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in VALID_SUFFIXES:
            yield file_path


def compute_sha256(file_path: str | Path) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def is_valid_image(file_path: str | Path, min_size: int = 64) -> bool:
    try:
        with Image.open(file_path) as image:
            image.load()
            image = image.convert("RGB")
            width, height = image.size
            return width >= min_size and height >= min_size
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def clean_and_deduplicate_images(directory: str | Path) -> dict[str, int]:
    directory = Path(directory)
    seen_hashes = set()
    invalid_count = 0
    duplicate_count = 0
    valid_count = 0

    for file_path in iter_image_files(directory):
        if not is_valid_image(file_path):
            file_path.unlink(missing_ok=True)
            invalid_count += 1
            continue

        file_hash = compute_sha256(file_path)
        if file_hash in seen_hashes:
            file_path.unlink(missing_ok=True)
            duplicate_count += 1
            continue

        seen_hashes.add(file_hash)
        valid_count += 1

    return {
        "valid_images": valid_count,
        "invalid_removed": invalid_count,
        "duplicate_removed": duplicate_count,
    }


def write_json(data: dict, file_path: str | Path) -> None:
    Path(file_path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
