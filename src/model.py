import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from typing import Optional, Tuple
import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 38
CHECKPOINTS = Path("checkpoints")



# EfficientNetPlant
class EfficientNetPlant(nn.Module):
    """
    Классификатор болезней растений на базе EfficientNet-B0.

    EfficientNet-B0 выбран потому что:
        - Обучен на ImageNet (1.2M изображений, 1000 классов)
        - Переносит знания о текстурах, краях, цветах → идеально для листьев
        - Лёгкий: 5.3M параметров (vs ResNet50: 25M)
        - Быстрый инференс: ~10ms на CPU (важно для веб-приложения)
        - Val accuracy: ~92-95% после fine-tuning

    Архитектура:
        EfficientNet-B0 backbone (features extractor)
            ↓ [B, 1280]  — выходной вектор фичей
        Custom head:
            Dropout(0.4)
            Linear(1280 → 512) + ReLU
            Dropout(0.2)
            Linear(512 → 38)
            ↓ [B, 38]  — логиты классов

    Двухфазное обучение:
        Фаза 1: freeze_backbone()
            - Обучаем только head
            - lr=1e-3, AdamW, CosineAnnealingLR
            - ~10 эпох, быстрая сходимость

        Фаза 2: unfreeze_backbone()
            - Обучаем всю сеть
            - lr=1e-4 (меньший lr — не ломаем pretrained веса)
            - AdamW, CosineAnnealingLR
            - ~15 эпох, финальная точность

    Args:
        num_classes : количество классов (default 38)
        dropout     : dropout в head (default 0.4)
        pretrained  : ImageNet веса (default True)
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.4,
        pretrained: bool = True,
    ):
        super().__init__()

        #  Backbone
        weights = "DEFAULT" if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)

        # Размер выходного вектора фичей EfficientNet-B0 = 1280
        self.in_features = backbone.classifier[1].in_features  # 1280

        # Убираем оригинальный classifier
        backbone.classifier = nn.Identity()
        self.backbone = backbone

        #  Custom Head 
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(512, num_classes),
        )

        # Инициализация head
        self._init_head()

        # Стартуем с замороженным backbone (Фаза 1)
        self.freeze_backbone()

    def _init_head(self):
        """Xavier инициализация для head."""
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def freeze_backbone(self):
        """
        Фаза 1 — замораживаем backbone.
        Обучаем только head.
        Используй: lr=1e-3, ~10 эпох.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False

        trainable = self._count_trainable()
        print(f"🔒 Backbone frozen")
        print(f"   Trainable params: {trainable:,} (head only)")

    def unfreeze_backbone(self):
        """
        Фаза 2 — размораживаем весь backbone.
        Full fine-tuning.
        Используй: lr=1e-4, ~15 эпох.
        ⚠️  Вызывать только после Фазы 1!
        """
        for param in self.backbone.parameters():
            param.requires_grad = True

        trainable = self._count_trainable()
        print(f"🔓 Full unfreeze")
        print(f"   Trainable params: {trainable:,} (full model)")

    def _count_trainable(self) -> int:
        return sum(
            p.numel() for p in self.parameters()
            if p.requires_grad
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x : [B, 3, 224, 224] нормализованный тензор

        Returns:
            logits : [B, 38] — сырые логиты (без softmax)
                     softmax применяется в inference.py
        """
        features = self.backbone(x)   # [B, 1280]
        logits   = self.head(features) # [B, 38]
        return logits

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Инференс — возвращает класс и вероятности.
        Используется в app/inference.py.

        Args:
            x : [B, 3, 224, 224]

        Returns:
            pred_classes : [B] — индексы классов
            probs        : [B, 38] — вероятности softmax
        """
        self.eval()
        with torch.no_grad():
            logits       = self.forward(x)
            probs        = torch.softmax(logits, dim=1)
            pred_classes = probs.argmax(dim=1)
        return pred_classes, probs



# FACTORY
def build_model(
    device: Optional[torch.device] = None,
    pretrained: bool = True,
    num_classes: int = NUM_CLASSES,
) -> EfficientNetPlant:
    """
    Создаёт EfficientNetPlant и переносит на device.

    Args:
        device      : torch.device (если None — автовыбор)
        pretrained  : использовать ImageNet веса
        num_classes : количество классов

    Returns:
        model : EfficientNetPlant на нужном device

    Example:
        model = build_model()
        # Фаза 1
        optimizer = AdamW(model.parameters(), lr=1e-3)
        train(model, ..., epochs=10)
        # Фаза 2
        model.unfreeze_backbone()
        optimizer = AdamW(model.parameters(), lr=1e-4)
        train(model, ..., epochs=15)
    """
    from src.preprocessing import get_device

    if device is None:
        device = get_device()

    model = EfficientNetPlant(
        num_classes=num_classes,
        pretrained=pretrained,
    )
    model = model.to(device)

    total = sum(p.numel() for p in model.parameters())
    print(f"\n🌿 EfficientNet-B0 (Transfer Learning)")
    print(f"   Backbone       : EfficientNet-B0 (ImageNet)")
    print(f"   Total params   : {total:,}")
    print(f"   Device         : {device}")
    print(f"   Num classes    : {num_classes}")

    return model



# SAVE / LOAD
def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_acc: float,
    model_name: str = "efficientnet_plant",
):
    """
    Сохраняет checkpoint.

    Файл: checkpoints/{model_name}_best.pth
    Содержит: веса, optimizer state, epoch, val_acc
    """
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch"      : epoch,
        "model_name" : model_name,
        "state_dict" : model.state_dict(),
        "optimizer"  : optimizer.state_dict(),
        "val_acc"    : val_acc,
        "num_classes": NUM_CLASSES,
    }

    path = CHECKPOINTS / f"{model_name}_best.pth"
    torch.save(state, path)
    print(f"💾 Saved → {path}  (val_acc={val_acc:.4f})")


def load_checkpoint(
    checkpoint_path: Path,
    device: Optional[torch.device] = None,
    pretrained: bool = False,
) -> Tuple[EfficientNetPlant, int, float]:
    from src.preprocessing import get_device

    if device is None:
        device = get_device()

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    # Достаём state_dict независимо от формата
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict  = checkpoint["state_dict"]
        epoch       = checkpoint.get("epoch", 0)
        val_acc     = checkpoint.get("val_acc", 0.0)
        num_classes = checkpoint.get("num_classes", NUM_CLASSES)
    else:
        # Чекпоинт — это сырой OrderedDict с весами
        state_dict  = checkpoint
        epoch       = 0
        val_acc     = 0.0
        num_classes = NUM_CLASSES

    model = EfficientNetPlant(num_classes=num_classes, pretrained=False)
    model.unfreeze_backbone()

    # Ремаппинг ключей: features.X → backbone.features.X
    #                   classifier.X → head.X
    remapped = {}
    for k, v in state_dict.items():
        if k.startswith("features.") or k.startswith("classifier."):
            new_key = "backbone." + k
            remapped[new_key] = v
        else:
            remapped[k] = v

    # Загружаем с strict=False чтобы не падало на несовпадающих ключах
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    print(f"   Missing keys  : {len(missing)}")
    print(f"   Unexpected keys: {len(unexpected)}")

    model = model.to(device)
    model.eval()

    print(f"✅ Model loaded: {checkpoint_path}")
    print(f"   Epoch  : {epoch}")
    print(f"   Val acc: {val_acc:.4f}")

    return model, epoch, val_acc