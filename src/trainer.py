import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import time
import copy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

CHECKPOINTS = Path("checkpoints")



# METRICS
class MetricsTracker:
    """
    Хранит историю метрик за все эпохи.
    Используется для графиков и early stopping.
    """

    def __init__(self):
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc":  [],
            "val_loss":   [],
            "val_acc":    [],
        }
        self.best_val_acc  = 0.0
        self.best_epoch    = 0

    def update(self, train_loss: float, train_acc: float,
               val_loss: float, val_acc: float, epoch: int):
        self.history["train_loss"].append(train_loss)
        self.history["train_acc"].append(train_acc)
        self.history["val_loss"].append(val_loss)
        self.history["val_acc"].append(val_acc)

        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_epoch   = epoch

    def print_epoch(self, epoch: int, total_epochs: int, elapsed: float):
        tl = self.history["train_loss"][-1]
        ta = self.history["train_acc"][-1]
        vl = self.history["val_loss"][-1]
        va = self.history["val_acc"][-1]
        best = "⭐" if va == self.best_val_acc else "  "

        print(
            f"Epoch [{epoch:>3}/{total_epochs}] "
            f"| Train loss: {tl:.4f}  acc: {ta:.4f} "
            f"| Val loss: {vl:.4f}  acc: {va:.4f} "
            f"| {elapsed:.1f}s {best}"
        )



# EARLY STOPPING
class EarlyStopping:
    """
    Останавливает обучение если val_loss не улучшается.

    Args:
        patience : сколько эпох ждать улучшения
        delta    : минимальное улучшение которое считается прогрессом
    """

    def __init__(self, patience: int = 7, delta: float = 1e-4):
        self.patience   = patience
        self.delta      = delta
        self.counter    = 0
        self.best_loss  = np.inf
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                print(f"\n⏹️  Early stopping triggered "
                      f"(no improvement for {self.patience} epochs)")
        return self.should_stop



# ONE EPOCH
def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    phase: str = "train",
) -> Tuple[float, float]:
    """
    Прогоняет одну эпоху train или val.

    Args:
        model     : модель
        loader    : DataLoader
        criterion : loss функция
        optimizer : None для val/test фазы
        device    : torch.device
        phase     : "train" | "val" | "test"

    Returns:
        avg_loss, avg_accuracy
    """
    is_train = (phase == "train")
    model.train() if is_train else model.eval()

    total_loss    = 0.0
    total_correct = 0
    total_samples = 0

    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        pbar = tqdm(loader, desc=f"  {phase:5s}",
                    leave=False, ncols=80)

        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Forward
            logits = model(images)
            loss   = criterion(logits, labels)

            # Backward (только train)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping — защита от взрыва градиентов
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # Метрики
            preds          = logits.argmax(dim=1)
            correct        = (preds == labels).sum().item()
            total_loss    += loss.item() * images.size(0)
            total_correct += correct
            total_samples += images.size(0)

            # Прогресс бар
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc":  f"{correct / images.size(0):.4f}",
            })

    avg_loss = total_loss / total_samples
    avg_acc  = total_correct / total_samples
    return avg_loss, avg_acc



# TRAINER
class Trainer:
    """
    Главный класс для обучения модели.

    Включает:
        - train / val loop
        - early stopping
        - сохранение лучшей модели
        - история метрик

    Args:
        model       : nn.Module
        dataloaders : {"train": ..., "val": ...}
        criterion   : loss функция
        optimizer   : оптимизатор
        scheduler   : lr scheduler (опционально)
        device      : torch.device
        model_name  : имя для сохранения checkpoint
        patience    : early stopping patience
    """

    def __init__(
        self,
        model: nn.Module,
        dataloaders: Dict[str, DataLoader],
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        model_name: str = "model",
        scheduler=None,
        patience: int = 7,
    ):
        self.model       = model
        self.dataloaders = dataloaders
        self.criterion   = criterion
        self.optimizer   = optimizer
        self.scheduler   = scheduler
        self.device      = device
        self.model_name  = model_name

        self.tracker      = MetricsTracker()
        self.early_stop   = EarlyStopping(patience=patience)
        self.best_weights = None

    def fit(self, epochs: int = 30) -> MetricsTracker:
        """
        Запускает обучение на epochs эпох.

        Returns:
            MetricsTracker с историей метрик
        """
       
        print(f"  Training: {self.model_name}")
        print(f"  Epochs  : {epochs} | Device: {self.device}")
       

        CHECKPOINTS.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            # Train
            train_loss, train_acc = run_epoch(
                self.model, self.dataloaders["train"],
                self.criterion, self.optimizer,
                self.device, phase="train",
            )

            # Validation
            val_loss, val_acc = run_epoch(
                self.model, self.dataloaders["val"],
                self.criterion, None,
                self.device, phase="val",
            )

            elapsed = time.time() - t0

            # Обновляем историю
            self.tracker.update(train_loss, train_acc,
                                 val_loss, val_acc, epoch)
            self.tracker.print_epoch(epoch, epochs, elapsed)

            # LR Scheduler
            if self.scheduler:
                if isinstance(self.scheduler,
                               torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Сохраняем лучшую модель
            is_best = (val_acc == self.tracker.best_val_acc)
            if is_best:
                self.best_weights = copy.deepcopy(
                    self.model.state_dict()
                )
                self._save_checkpoint(epoch, val_acc, is_best=True)

            # Early stopping
            if self.early_stop(val_loss):
                break

        # Восстанавливаем лучшие веса
        if self.best_weights:
            self.model.load_state_dict(self.best_weights)

        print(f"\n✅ Training complete!")
        print(f"   Best val_acc : {self.tracker.best_val_acc:.4f} "
              f"(epoch {self.tracker.best_epoch})")

        return self.tracker

    def evaluate(self, phase: str = "test") -> Tuple[float, float]:
        """
        Оценивает модель на test или val наборе.

        Returns:
            loss, accuracy
        """
        if phase not in self.dataloaders:
            print(f"⚠️  No '{phase}' dataloader found")
            return 0.0, 0.0

        loss, acc = run_epoch(
            self.model, self.dataloaders[phase],
            self.criterion, None,
            self.device, phase=phase,
        )
        print(f"\n📊 {phase.upper()} Results:")
        print(f"   Loss     : {loss:.4f}")
        print(f"   Accuracy : {acc:.4f} ({acc*100:.2f}%)")
        return loss, acc

    def _save_checkpoint(self, epoch: int, val_acc: float,
                          is_best: bool = False):
        state = {
            "epoch"     : epoch,
            "model_name": self.model_name,
            "state_dict": self.model.state_dict(),
            "optimizer" : self.optimizer.state_dict(),
            "val_acc"   : val_acc,
        }
        path = CHECKPOINTS / f"{self.model_name}_last.pth"
        torch.save(state, path)

        if is_best:
            best = CHECKPOINTS / f"{self.model_name}_best.pth"
            torch.save(state, best)



# OPTIMIZER & SCHEDULER FACTORY
def build_optimizer(
    model: nn.Module,
    optimizer_name: str = "adam",
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    """
    Adam  — быстрая сходимость, хорош для Transfer Learning
    SGD   — медленнее, но часто лучший результат для CNN with scratch
    AdamW — Adam + правильный weight decay (рекомендую для TL)
    """
    params = filter(lambda p: p.requires_grad, model.parameters())

    if optimizer_name == "adam":
        return torch.optim.Adam(params, lr=lr,
                                 weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        return torch.optim.AdamW(params, lr=lr,
                                  weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9,
                                weight_decay=weight_decay,
                                nesterov=True)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str = "reduce_on_plateau",
    epochs: int = 30,
):
    """
    reduce_on_plateau — уменьшает lr если val_loss не улучшается
    cosine            — плавное уменьшение lr по косинусу
    step              — уменьшает lr каждые N эпох
    """
    if scheduler_name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5,
            patience=3, 
        )
    elif scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6,
        )
    elif scheduler_name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.1,
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
