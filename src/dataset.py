import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from src.preprocessing import get_transforms, IMAGENET_MEAN , IMAGENET_STD

# ImageNet stats — стандарт для Transfer Learning
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png",
                    ".JPG", ".JPEG", ".PNG"}

# Пути по умолчанию
DATA_ROOT    = Path("data/raw")
PROCESSED    = Path("data/processed")
CHECKPOINTS  = Path("checkpoints")



# CLASS MAPPING
def build_class_mapping(train_dir: Path) -> Tuple[Dict, Dict]:
    """
    Строит маппинг классов из папки train.

    Returns:
        class_to_idx : {"Apple___scab": 0, ...}
        idx_to_class : {0: "Apple___scab", ...}
    """
    classes = sorted([
        d.name for d in Path(train_dir).iterdir()
        if d.is_dir()
    ])
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    return class_to_idx, idx_to_class


def save_class_mapping(class_to_idx: Dict, idx_to_class: Dict,
                        save_dir: Path = PROCESSED):
    """Сохраняет маппинг в JSON — нужен для инференса в продакшене."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "class_to_idx": class_to_idx,
        "idx_to_class": {str(k): v for k, v in idx_to_class.items()},
        "num_classes": len(class_to_idx),
    }
    out = save_dir / "class_mapping.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"✅ Class mapping saved → {out}")


def load_class_mapping(path: Path = PROCESSED / "class_mapping.json"
                        ) -> Tuple[Dict, Dict]:
    """Загружает маппинг из JSON."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    class_to_idx = data["class_to_idx"]
    idx_to_class = {int(k): v for k, v in data["idx_to_class"].items()}
    return class_to_idx, idx_to_class



# DATASET
class PlantDiseaseDataset(Dataset):
    """
    Dataset для New Plant Diseases формата.

    Структура папки:
        root_dir/
            Apple___Apple_scab/
                img001.jpg
                ...
            Apple___healthy/
                img001.jpg
            ...

    Args:
        root_dir     : путь до папки с классами (train / valid / test)
        class_to_idx : маппинг {class_name: int}
        transform    : torchvision transforms
        return_path  : если True → возвращает (image, label, path)
    """

    def __init__(
        self,
        root_dir: Path,
        class_to_idx: Dict[str, int],
        transform: Optional[transforms.Compose] = None,
        return_path: bool = False,
    ):
        self.root_dir     = Path(root_dir)
        self.class_to_idx = class_to_idx
        self.transform    = transform
        self.return_path  = return_path
        self.samples      = self._load_samples()

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found in {self.root_dir}\n"
                f"Check path and folder structure."
            )

    def _load_samples(self) -> List[Tuple[str, int]]:
        samples = []
        for class_name, label in self.class_to_idx.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix in VALID_EXTENSIONS:
                    samples.append((str(img_path), label))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        if self.return_path:
            return image, label, img_path
        return image, label

    def get_class_weights(self) -> torch.Tensor:
        """
        Веса классов для борьбы с дисбалансом.
        Используется в CrossEntropyLoss(weight=...).
        """
        counts = np.zeros(len(self.class_to_idx))
        for _, label in self.samples:
            counts[label] += 1
        weights = 1.0 / (counts + 1e-6)
        weights = weights / weights.sum() * len(self.class_to_idx)
        return torch.FloatTensor(weights)

    def get_labels(self) -> List[int]:
        """Возвращает все лейблы — нужен для confusion matrix."""
        return [label for _, label in self.samples]



# DATALOADER FACTORY

def get_dataloaders(
    class_to_idx: Dict[str, int],
    data_root: Path = DATA_ROOT,
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 2,
) -> Tuple[Dict[str, DataLoader], Dict[str, PlantDiseaseDataset]]:
    """
    Создаёт DataLoader'ы для train / val / test.

    Args:
        class_to_idx : маппинг классов
        data_root    : корень с папками train/ valid/ test/
        img_size     : размер изображения (224 стандарт)
        batch_size   : размер батча
        num_workers  : потоки (2 для Mac M-series)

    Returns:
        dataloaders, datasets
    """

    

    data_root  = Path(data_root)
    phase_dirs = {
        "train" : data_root / "train",
        "val"   : data_root / "valid",
        "test"  : data_root / "test",
    }

    datasets    = {}
    dataloaders = {}

    for phase, folder in phase_dirs.items():
        if not folder.exists():
            print(f"⚠️  Skipping '{phase}' — not found: {folder}")
            continue

        dataset = PlantDiseaseDataset(
            root_dir=folder,
            class_to_idx=class_to_idx,
            transform=get_transforms(phase, img_size),
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(phase == "train"),
            num_workers=num_workers,
            pin_memory=False,   # False для Apple Silicon MPS
            drop_last=(phase == "train"),
        )

        datasets[phase]    = dataset
        dataloaders[phase] = loader

        n_batches = len(loader)
        print(f"  {phase:5s} → {len(dataset):>7,} images "
              f"| {n_batches:>4} batches "
              f"| folder: {folder.name}/")

    return dataloaders, datasets
