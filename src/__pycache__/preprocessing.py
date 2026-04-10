import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))


from pathlib import Path
from typing import Dict, Tuple

import torch
import numpy as np
from torchvision import transforms


# ImageNet stats — стандарт для Transfer Learning
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

IMG_SIZE = 224  # стандарт для ResNet / EfficientNet



# TRANSFORMS

def get_transforms(phase: str, img_size: int = IMG_SIZE) -> transforms.Compose:
    """
    Возвращает transforms для нужной фазы.

    Train  → аугментация + normalize
    Val    → только resize + normalize
    Test   → только resize + normalize
    Infer  → только resize + normalize (продакшен)

    Args:
        phase    : "train" | "val" | "test" | "infer"
        img_size : размер выходного изображения (default 224)
    """
    assert phase in {"train", "val", "test", "infer"}, \
        f"Unknown phase '{phase}'. Use: train / val / test / infer"

    if phase == "train":
        return transforms.Compose([
            # 1. Чуть больше → random crop → избегаем артефактов по краям
            transforms.Resize((img_size + 24, img_size + 24)),
            transforms.RandomCrop(img_size),

            # 2. Геометрические аугментации
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=20),

            # 3. Цветовые аугментации — важно для болезней листьев
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.05,
            ),
            transforms.RandomGrayscale(p=0.05),

            # 4. Tensor + Normalize
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    else:  # val / test / infer
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])



# DENORMALIZE — для визуализации
def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Обратная нормализация для отображения изображений.
    Использовать перед imshow().

    Args:
        tensor : нормализованный тензор [C, H, W] или [B, C, H, W]

    Returns:
        тензор в диапазоне [0, 1]
    """
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    if tensor.dim() == 4:  # batch
        mean = mean.unsqueeze(0)
        std  = std.unsqueeze(0)

    return (tensor * std + mean).clamp(0, 1)



# CLASS WEIGHTS — борьба с дисбалансом (68% diseased / 32% healthy)

def compute_class_weights(
    class_counts: Dict[str, int],
    class_to_idx: Dict[str, int],
    device: torch.device,
) -> torch.Tensor:
    """
    Считает веса классов для CrossEntropyLoss.
    Нужно потому что датасет несбалансирован:
        Healthy  : 31.7%
        Diseased : 68.3%

    Args:
        class_counts : {"Apple___scab": 630, ...}
        class_to_idx : {"Apple___scab": 0, ...}
        device       : torch.device

    Returns:
        weights тензор shape [num_classes]
    """
    num_classes = len(class_to_idx)
    counts = np.zeros(num_classes)

    for class_name, idx in class_to_idx.items():
        counts[idx] = class_counts.get(class_name, 1)

    # Inverse frequency weighting
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes

    return torch.FloatTensor(weights).to(device)


def get_class_counts(data_dir: Path, class_to_idx: Dict[str, int]) -> Dict[str, int]:
    """
    Считает количество изображений в каждом классе.

    Args:
        data_dir     : путь до train папки
        class_to_idx : маппинг классов

    Returns:
        {"Apple___scab": 630, ...}
    """
    counts = {}
    for class_name in class_to_idx:
        class_dir = Path(data_dir) / class_name
        if class_dir.exists():
            n = len([
                f for f in class_dir.iterdir()
                if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ])
            counts[class_name] = n
        else:
            counts[class_name] = 0
    return counts



# DEVICE SETUP
def get_device() -> torch.device:
    """
    Автоматически выбирает лучшее устройство.
    Apple Silicon → MPS
    NVIDIA        → CUDA
    Otherwise     → CPU
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🖥️  Device: Apple Silicon MPS")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🖥️  Device: CUDA ({gpu_name})")
    else:
        device = torch.device("cpu")
        print("🖥️  Device: CPU")
    return device



# QUICK TEST

if __name__ == "__main__":
    from PIL import Image
    import matplotlib.pyplot as plt

    print("=" * 55)
    print("  PlantGuard AI — preprocessing.py test")
    print("=" * 55)

    # 1. Device
    device = get_device()

    # 2. Transforms
    for phase in ["train", "val", "test", "infer"]:
        t = get_transforms(phase)
        print(f"\n✅ Transform [{phase}]:")
        for step in t.transforms:
            print(f"   {step}")

    # 3. Тест денормализации
    dummy = torch.randn(3, 224, 224)
    train_t = get_transforms("train")

    # Берём реальное изображение если есть
    test_path = Path("data/raw/train")
    sample_img = None
    for class_dir in test_path.iterdir():
        if class_dir.is_dir():
            imgs = list(class_dir.iterdir())
            if imgs:
                sample_img = imgs[0]
                break

    if sample_img:
        img = Image.open(sample_img).convert("RGB")
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        # Оригинал
        axes[0].imshow(img.resize((224, 224)))
        axes[0].set_title("Original")
        axes[0].axis("off")

        # Аугментированные
        for i, ax in enumerate(axes[1:], 1):
            aug = train_t(img)
            vis = denormalize(aug).permute(1, 2, 0).numpy()
            ax.imshow(vis)
            ax.set_title(f"Augmented #{i}")
            ax.axis("off")

        plt.suptitle("Preprocessing Test", fontweight="bold")
        plt.tight_layout()
        plt.savefig("notebooks/plots/preprocessing_test.png", dpi=120)
        plt.show()
        print("\n✅ Визуализация сохранена → notebooks/plots/preprocessing_test.png")

    # 4. Класс веса
    print("\n✅ Class weights test:")
    dummy_counts  = {"Apple___scab": 630, "Apple___healthy": 1645}
    dummy_mapping = {"Apple___scab": 0,   "Apple___healthy": 1}
    weights = compute_class_weights(dummy_counts, dummy_mapping, torch.device("cpu"))
    print(f"   Weights: {weights.tolist()}")

    print("\n✅ preprocessing.py — ALL GOOD\n")