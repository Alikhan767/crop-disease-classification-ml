import sys
from pathlib import Path

# app/backend/inference.py → нужно подняться на 3 уровня
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple

from src.model import EfficientNetPlant, load_checkpoint
from src.preprocessing import get_transforms, get_device


# DISEASE INFO — описание болезней для пользователя
DISEASE_INFO: Dict[str, Dict] = {
    "healthy": {
        "severity"   : "none",
        "description": "Plant is healthy. No disease detected.",
        "treatment"  : "No treatment needed. Keep monitoring regularly.",
    },

    # APPLE
    "Apple_scab": {
        "severity"   : "moderate",
        "description": "Fungal disease causing dark scabby lesions on leaves and fruit.",
        "treatment"  : "Spray with fungicide (myclobutanil or captan). Remove fallen leaves. Prune for air circulation.",
    },
    "Black_rot": {
        "severity"   : "high",
        "description": "Fungal disease causing black rotting on fruits and leaves.",
        "treatment"  : "Remove infected fruit and branches. Apply copper-based fungicide. Avoid overhead watering.",
    },
    "Cedar_apple_rust": {
        "severity"   : "moderate",
        "description": "Fungal disease causing orange-yellow spots on leaves.",
        "treatment"  : "Apply fungicide (myclobutanil) in spring. Remove nearby juniper/cedar trees if possible.",
    },

    # CHERRY
    "Powdery_mildew": {
        "severity"   : "moderate",
        "description": "Fungal disease causing white powdery coating on leaves.",
        "treatment"  : "Apply sulfur-based fungicide. Improve air circulation. Avoid excess nitrogen fertilizer.",
    },

    # CORN
    "Cercospora_leaf_spot": {
        "severity"   : "moderate",
        "description": "Fungal disease causing gray leaf spots on corn.",
        "treatment"  : "Apply fungicide (strobilurin). Rotate crops. Use resistant varieties.",
    },
    "Common_rust": {
        "severity"   : "moderate",
        "description": "Fungal disease causing rust-colored pustules on corn leaves.",
        "treatment"  : "Apply fungicide early. Plant resistant hybrids. Monitor fields regularly.",
    },
    "Northern_Leaf_Blight": {
        "severity"   : "high",
        "description": "Fungal disease causing large tan lesions on corn leaves.",
        "treatment"  : "Apply fungicide (propiconazole). Rotate crops. Use resistant varieties.",
    },

    # GRAPE
    "Black_measles": {
        "severity"   : "high",
        "description": "Fungal disease causing dark spots and vine decline.",
        "treatment"  : "Prune infected wood. Apply fungicide. Avoid water stress.",
    },
    "Leaf_blight": {
        "severity"   : "moderate",
        "description": "Fungal disease causing brown leaf margins and defoliation.",
        "treatment"  : "Apply copper fungicide. Remove infected leaves. Improve drainage.",
    },
    "Esca": {
        "severity"   : "high",
        "description": "Complex fungal disease causing wood decay in grapevines.",
        "treatment"  : "Remove and destroy infected vines. Apply fungicide to pruning wounds.",
    },

    # POTATO
    "Early_blight": {
        "severity"   : "moderate",
        "description": "Fungal disease causing brown spots with concentric rings.",
        "treatment"  : "Apply fungicide (chlorothalonil). Remove infected leaves. Rotate crops.",
    },
    "Late_blight": {
        "severity"   : "high",
        "description": "Water mold causing rapid dark lesions and plant decay.",
        "treatment"  : "Apply fungicide immediately (metalaxyl). Remove infected plants. Avoid overhead irrigation.",
    },

    # TOMATO
    "Bacterial_spot": {
        "severity"   : "moderate",
        "description": "Bacterial disease causing water-soaked spots on leaves and fruit.",
        "treatment"  : "Apply copper bactericide. Remove infected material. Avoid working with wet plants.",
    },
    "Leaf_Mold": {
        "severity"   : "moderate",
        "description": "Fungal disease causing yellow spots and gray mold on leaves.",
        "treatment"  : "Improve ventilation. Apply fungicide. Reduce humidity in greenhouse.",
    },
    "Septoria_leaf_spot": {
        "severity"   : "moderate",
        "description": "Fungal disease causing small circular spots with dark borders.",
        "treatment"  : "Apply fungicide (chlorothalonil). Remove lower infected leaves. Avoid overhead watering.",
    },
    "Spider_mites": {
        "severity"   : "moderate",
        "description": "Pest causing yellowing and bronzing of leaves.",
        "treatment"  : "Apply miticide or neem oil. Increase humidity. Introduce predatory mites.",
    },
    "Target_Spot": {
        "severity"   : "moderate",
        "description": "Fungal disease causing concentric ring spots on leaves.",
        "treatment"  : "Apply fungicide (azoxystrobin). Remove infected leaves. Improve air circulation.",
    },
    "Tomato_Yellow_Leaf_Curl_Virus": {
        "severity"   : "high",
        "description": "Viral disease spread by whiteflies causing leaf curl and yellowing.",
        "treatment"  : "Control whiteflies with insecticide. Remove infected plants. Use virus-resistant varieties.",
    },
    "Tomato_mosaic_virus": {
        "severity"   : "high",
        "description": "Viral disease causing mosaic patterns and distorted leaves.",
        "treatment"  : "Remove infected plants. Disinfect tools. Control aphids. No chemical cure.",
    },

    # STRAWBERRY
    "Leaf_scorch": {
        "severity"   : "moderate",
        "description": "Fungal disease causing purple-red spots and leaf scorch.",
        "treatment"  : "Apply fungicide. Remove infected leaves. Avoid overhead watering.",
    },

    # SQUASH
    "Powdery_mildew_squash": {
        "severity"   : "moderate",
        "description": "Fungal disease causing white powder on squash leaves.",
        "treatment"  : "Apply potassium bicarbonate or sulfur fungicide. Improve air flow.",
    },

    "default": {
        "severity"   : "unknown",
        "description": "Disease detected. Consult an agronomist for accurate diagnosis.",
        "treatment"  : "Apply appropriate fungicide or pesticide. Remove infected parts. Consult local extension service.",
    },
}

SEVERITY_COLORS = {
    "none"    : "#2d9e75",  # зелёный
    "moderate": "#f5a623",  # оранжевый
    "high"    : "#e8593c",  # красный
    "unknown" : "#888888",  # серый
}



# PREDICTOR
class PlantDiseasePredictor:
    """
    Класс для инференса модели в продакшене.

    Использование:
        predictor = PlantDiseasePredictor()
        result    = predictor.predict(image_path)

    Args:
        checkpoint_path : путь к .pth файлу
        mapping_path    : путь к class_mapping.json
        device          : torch.device (если None — автовыбор)
        top_k           : сколько топ предсказаний возвращать
    """

    def __init__(
        self,
        checkpoint_path: Path = Path('/Users/shopangaliy/Documents/Doc/IT3-2208/4 course/2_semester/DeepLearning_Ai/PlantGuard_AI /checkpoints/efficientnet_b0_best.pth'),
        mapping_path: Path    = Path("data/processed/class_mapping.json"),
        device: torch.device  = None,
        top_k: int            = 3,
    ):
        self.device   = device or get_device()
        self.top_k    = top_k
        self.transform = get_transforms("infer")

        # Загружаем маппинг классов
        self.class_to_idx, self.idx_to_class = \
            self._load_mapping(mapping_path)

        # Загружаем модель
        self.model, self.epoch, self.val_acc = \
            load_checkpoint(checkpoint_path, self.device)

        print(f"\n✅ Predictor ready")
        print(f"   Classes : {len(self.idx_to_class)}")
        print(f"   Val acc : {self.val_acc:.4f} ({self.val_acc*100:.2f}%)")

    def _load_mapping(self, path: Path) -> Tuple[Dict, Dict]:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        class_to_idx = data["class_to_idx"]
        idx_to_class = {
            int(k): v for k, v in data["idx_to_class"].items()
        }
        return class_to_idx, idx_to_class

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        """PIL Image → нормализованный тензор [1, 3, 224, 224]."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        tensor = self.transform(image)
        return tensor.unsqueeze(0).to(self.device)

    def _parse_class_name(self, class_name: str) -> Tuple[str, str]:
        """
        "Tomato___Late_blight" → ("Tomato", "Late blight")
        "Apple___healthy"      → ("Apple", "Healthy")
        """
        parts   = class_name.split("___")
        plant   = parts[0].replace("_", " ").strip()
        disease = parts[1].replace("_", " ").strip() \
            if len(parts) > 1 else "Unknown"
        return plant, disease

    def _get_disease_info(self, disease: str) -> Dict:
        """Возвращает информацию о болезни."""
        for key in DISEASE_INFO:
            if key.lower() in disease.lower():
                return DISEASE_INFO[key]
        return DISEASE_INFO["default"]

    def predict(self, image: Image.Image) -> Dict:
        """
        Основной метод предсказания.

        Args:
            image : PIL Image

        Returns:
            {
                "plant"       : "Tomato",
                "disease"     : "Late blight",
                "is_healthy"  : False,
                "confidence"  : 0.9823,
                "severity"    : "high",
                "description" : "...",
                "treatment"   : "...",
                "color"       : "#e8593c",
                "top_k"       : [
                    {"class": "Tomato___Late_blight",
                     "plant": "Tomato",
                     "disease": "Late blight",
                     "confidence": 0.9823},
                    ...
                ]
            }
        """
        # Препроцессинг
        tensor = self._preprocess(image)

        # Инференс
        pred_classes, probs = self.model.predict(tensor)

        # Топ-K предсказаний
        top_probs, top_indices = torch.topk(
            probs[0], k=self.top_k
        )

        # Основное предсказание
        top_class  = self.idx_to_class[top_indices[0].item()]
        confidence = top_probs[0].item()
        plant, disease = self._parse_class_name(top_class)

        is_healthy   = "healthy" in top_class.lower()
        disease_info = self._get_disease_info(
            "healthy" if is_healthy else disease
        )

        # Топ-K список
        top_k_list = []
        for prob, idx in zip(top_probs, top_indices):
            cls_name    = self.idx_to_class[idx.item()]
            p, d        = self._parse_class_name(cls_name)
            top_k_list.append({
                "class"     : cls_name,
                "plant"     : p,
                "disease"   : d,
                "confidence": round(prob.item(), 4),
            })

        return {
            "plant"      : plant,
            "disease"    : disease,
            "is_healthy" : is_healthy,
            "confidence" : round(confidence, 4),
            "severity"   : disease_info["severity"],
            "description": disease_info["description"],
            "treatment"  : disease_info["treatment"],
            "color"      : SEVERITY_COLORS[disease_info["severity"]],
            "top_k"      : top_k_list,
        }

    def predict_from_path(self, image_path: Path) -> Dict:
        """Predict из файла."""
        image = Image.open(image_path).convert("RGB")
        return self.predict(image)

    def predict_from_bytes(self, image_bytes: bytes) -> Dict:
        """
        Predict из байтов.
        Используется в FastAPI endpoint.
        """
        import io
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return self.predict(image)
