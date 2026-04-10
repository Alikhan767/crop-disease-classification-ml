import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import io
import time
import logging
from typing import Dict, List, Optional

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image

from app.backend.inference import PlantDiseasePredictor


# LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# APP
app = FastAPI(
    title="PlantGuard AI",
    description="Plant Disease Detection API using EfficientNet-B0",
    version="1.0.0",
    docs_url="/docs",       # Swagger UI
    redoc_url="/redoc",     # ReDoc UI
)

# CORS — разрешаем React фронтенду обращаться к API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    
    allow_methods=["*"],
    allow_headers=["*"],
)


# ГЛОБАЛЬНЫЙ PREDICTOR — загружается один раз при старте
predictor: Optional[PlantDiseasePredictor] = None

CHECKPOINT_PATH = ROOT / "checkpoints" / "efficientnet_b0_phase2_last.pth"
MAPPING_PATH    = ROOT / "data" / "processed" / "class_mapping.json"

# Допустимые форматы изображений
ALLOWED_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@app.on_event("startup")
async def startup_event():
    """Загружаем модель при старте сервера."""
    global predictor
    logger.info("🚀 Starting PlantGuard AI server...")

    if not CHECKPOINT_PATH.exists():
        logger.warning(
            f"⚠️  Checkpoint not found: {CHECKPOINT_PATH}\n"
            f"   Скачай .pth с Kaggle и положи в checkpoints/"
        )
        return

    try:
        predictor = PlantDiseasePredictor(
            checkpoint_path=CHECKPOINT_PATH,
            mapping_path=MAPPING_PATH,
            top_k=3,
        )
        logger.info("✅ Model loaded successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")



# RESPONSE MODELS
class TopKPrediction(BaseModel):
    class_name : str
    plant      : str
    disease    : str
    confidence : float


class PredictionResponse(BaseModel):
    success    : bool
    plant      : str
    disease    : str
    is_healthy : bool
    confidence : float
    severity   : str
    description: str
    treatment  : str
    color      : str
    top_k      : List[TopKPrediction]
    time_ms    : float


class HealthResponse(BaseModel):
    status     : str
    model      : str
    val_acc    : Optional[float]
    device     : str
    checkpoint : bool



# ENDPOINTS
@app.get("/", tags=["Root"])
async def root():
    """Корневой эндпоинт."""
    return {
        "name"   : "PlantGuard AI",
        "version": "1.0.0",
        "status" : "running",
        "docs"   : "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """
    Проверка состояния сервера и модели.
    Используется для мониторинга в продакшене.
    """
    checkpoint_exists = CHECKPOINT_PATH.exists()

    if predictor is None:
        return HealthResponse(
            status     ="degraded",
            model      ="EfficientNet-B0",
            val_acc    =None,
            device     =str(torch.device("cpu")),
            checkpoint =checkpoint_exists,
        )

    return HealthResponse(
        status    ="healthy",
        model     ="EfficientNet-B0",
        val_acc   =predictor.val_acc,
        device    =str(predictor.device),
        checkpoint=checkpoint_exists,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(file: UploadFile = File(...)):
    """
    Основной эндпоинт — предсказание болезни растения.

    Принимает изображение листа растения.
    Возвращает название болезни, уверенность и рекомендации.

    - **file**: JPG / PNG / WEBP изображение (макс. 10 MB)
    """
    t0 = time.time()

    # Проверяем что модель загружена
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. "
                   "Place checkpoint in checkpoints/ and restart.",
        )

    # Проверяем тип файла
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. "
                   f"Allowed: {ALLOWED_TYPES}",
        )

    # Читаем файл
    image_bytes = await file.read()

    # Проверяем размер
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size: 10MB",
        )

    # Инференс
    try:
        result = predictor.predict_from_bytes(image_bytes)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )

    elapsed_ms = (time.time() - t0) * 1000

    logger.info(
        f"Predicted: {result['plant']} — {result['disease']} "
        f"({result['confidence']*100:.1f}%) | {elapsed_ms:.1f}ms"
    )

    # Форматируем top_k
    top_k_formatted = [
        TopKPrediction(
            class_name=p["class"],
            plant     =p["plant"],
            disease   =p["disease"],
            confidence=p["confidence"],
        )
        for p in result["top_k"]
    ]

    return PredictionResponse(
        success    =True,
        plant      =result["plant"],
        disease    =result["disease"],
        is_healthy =result["is_healthy"],
        confidence =result["confidence"],
        severity   =result["severity"],
        description=result["description"],
        treatment  =result["treatment"],
        color      =result["color"],
        top_k      =top_k_formatted,
        time_ms    =round(elapsed_ms, 2),
    )


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Batch предсказание — несколько изображений за раз.
    Максимум 10 изображений.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Max 10 images per batch",
        )

    results = []
    for file in files:
        try:
            image_bytes = await file.read()
            result      = predictor.predict_from_bytes(image_bytes)
            results.append({
                "filename"  : file.filename,
                "success"   : True,
                "plant"     : result["plant"],
                "disease"   : result["disease"],
                "is_healthy": result["is_healthy"],
                "confidence": result["confidence"],
                "severity"  : result["severity"],
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success" : False,
                "error"   : str(e),
            })

    return {"results": results, "total": len(results)}





if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,       # автоперезагрузка при изменении кода
        log_level="info",
    )