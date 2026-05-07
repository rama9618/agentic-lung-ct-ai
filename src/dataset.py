from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


CLASS_NAMES = (
    "adenocarcinoma",
    "large.cell.carcinoma",
    "normal",
    "squamous.cell.carcinoma",
)

CLASS_ALIASES = {
    "adenocarcinoma": "adenocarcinoma",
    "adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib": "adenocarcinoma",
    "large.cell.carcinoma": "large.cell.carcinoma",
    "large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa": "large.cell.carcinoma",
    "normal": "normal",
    "squamous.cell.carcinoma": "squamous.cell.carcinoma",
    "squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa": "squamous.cell.carcinoma",
}

IMAGE_EXTENSIONS = {
    ".bmp",
    ".dib",
    ".gif",
    ".jfif",
    ".jpe",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: int


def get_transforms(train: bool = False) -> transforms.Compose:
    steps = []

    if train:
        steps.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=10),
            ]
        )

    steps.extend(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    return transforms.Compose(steps)


class LungTumorDataset(Dataset):
    """PIL-backed image dataset for the lung tumor classification splits."""

    def __init__(self, root_dir: str | Path, split: str, transform=None) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.split_dir = self.root_dir / split
        self.transform = transform
        self.class_to_idx = {class_name: index for index, class_name in enumerate(CLASS_NAMES)}
        self.idx_to_class = {index: class_name for class_name, index in self.class_to_idx.items()}
        self.samples = self._find_samples()

    def _find_samples(self) -> list[ImageRecord]:
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        samples: list[ImageRecord] = []
        for class_dir in sorted(path for path in self.split_dir.iterdir() if path.is_dir()):
            class_name = CLASS_ALIASES.get(class_dir.name)
            if class_name is None:
                raise ValueError(f"Unknown class folder: {class_dir}")

            label = self.class_to_idx[class_name]
            for image_path in sorted(class_dir.rglob("*")):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    samples.append(ImageRecord(path=image_path, label=label))

        if not samples:
            raise ValueError(f"No images found in split directory: {self.split_dir}")

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        record = self.samples[index]
        with Image.open(record.path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, record.label


def create_dataset(data_dir: str | Path = "data/Data", split: str = "train") -> LungTumorDataset:
    return LungTumorDataset(
        root_dir=data_dir,
        split=split,
        transform=get_transforms(train=split == "train"),
    )


def create_dataloader(
    data_dir: str | Path = "data/Data",
    split: str = "train",
    batch_size: int = 32,
    shuffle: bool | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    dataset = create_dataset(data_dir=data_dir, split=split)
    should_shuffle = split == "train" if shuffle is None else shuffle
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=should_shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def create_dataloaders(
    data_dir: str | Path = "data/Data",
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> dict[str, DataLoader]:
    return {
        split: create_dataloader(
            data_dir=data_dir,
            split=split,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        for split in ("train", "valid", "test")
    }


if __name__ == "__main__":
    for split_name in ("train", "valid", "test"):
        dataset = create_dataset(split=split_name)
        print(f"{split_name}: {len(dataset)} images, classes={dataset.class_to_idx}")
