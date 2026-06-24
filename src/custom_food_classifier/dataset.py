from __future__ import annotations

from pathlib import Path

import torch
from PIL import ImageFile
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
ImageFile.LOAD_TRUNCATED_IMAGES = True


def build_transforms(image_size: int = 224):
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def build_dataloaders(
    dataset_root: str,
    batch_size: int = 32,
    num_workers: int = 2,
    image_size: int = 224,
):
    train_transform, eval_transform = build_transforms(image_size)
    dataset_root = Path(dataset_root)

    train_dataset = datasets.ImageFolder(dataset_root / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(dataset_root / "val", transform=eval_transform)
    test_dataset = datasets.ImageFolder(dataset_root / "test", transform=eval_transform)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader
