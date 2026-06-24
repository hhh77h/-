from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from custom_food_classifier.dataset import IMAGENET_MEAN, IMAGENET_STD
from custom_food_classifier.model import create_resnet18_classifier


def build_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="使用训练好的模型进行单图预测")
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--checkpoint", default="./scraped_food_data/outputs/best_model.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    bundle = torch.load(args.checkpoint, map_location=device)
    class_names = bundle["class_names"]
    image_size = bundle["image_size"]

    model = create_resnet18_classifier(num_classes=len(class_names), pretrained=False).to(device)
    model.load_state_dict(bundle["model_state_dict"])
    model.eval()

    image = Image.open(Path(args.image_path)).convert("RGB")
    tensor = build_transform(image_size)(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
        scores, indices = torch.topk(probabilities, k=args.top_k)

    for score, index in zip(scores.cpu().tolist(), indices.cpu().tolist()):
        print(f"{class_names[index]}: {score:.4f}")


if __name__ == "__main__":
    main()
