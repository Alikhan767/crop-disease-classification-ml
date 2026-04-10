# 🌿 PlantGuard AI

> Automatic classification of agricultural crops and early detection of plant diseases using computer vision and deep learning.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19.x-blue.svg)](https://react.dev)
[![Accuracy](https://img.shields.io/badge/Val%20Accuracy-99.83%25-brightgreen.svg)]()

---

## 📋 Overview

PlantGuard AI is a full-stack web application that allows farmers and agronomists to detect plant diseases from leaf photographs in real time. The system combines a high-accuracy EfficientNet-B0 deep learning model with an intuitive drag-and-drop web interface.

**Key results:**
- ✅ **99.83%** validation accuracy on 38 disease classes
- ✅ **~45ms** inference time on CPU
- ✅ **38 disease categories** across 14 plant species
- ✅ **87,867 training images** from New Plant Diseases Dataset

---

## 🏗️ Project Structure

```
PlantGuard_AI/
├── src/                        # Model source code
│   ├── model.py                # EfficientNetPlant class
│   ├── dataset.py              # PlantDiseaseDataset
│   ├── preprocessing.py        # Transforms and augmentation
│   └── trainer.py              # Training loop
│
├── app/
│   ├── backend/                # FastAPI backend
│   │   ├── main.py             # App entrypoint, routes
│   │   ├── inference.py        # PlantPredictor class
│   │   └── config.py           # Configuration
│   │
│   └── frontend/               # React frontend
│       ├── src/
│       │   ├── api/api.js      # HTTP client
│       │   ├── hooks/usePredict.js
│       │   └── components/
│       │       ├── Header.jsx
│       │       ├── DropZone.jsx
│       │       ├── ResultCard.jsx
│       │       ├── DiseaseInfo.jsx
│       │       └── TopKChart.jsx
│       ├── package.json
│       └── vite.config.js
│
├── notebooks/
│   ├── 01_EDA.ipynb            # Exploratory data analysis
│   ├── 02_Experiments.ipynb    # PlantCNN baseline experiments
│   └── plots/                  # Generated figures
│
├── checkpoints/
│   └── efficientnet_b0_phase2_last.pth   # Trained model
│
├── data/
│   ├── raw/
│   │   ├── train/              # 70,295 training images
│   │   └── valid/              # 17,572 validation images
│   └── processed/
│       └── class_mapping.json  # 38 class index mapping
│
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/PlantGuard_AI.git
cd PlantGuard_AI
```

### 2. Install Python dependencies

```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 3. Download the dataset

Download the [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) from Kaggle and place it in `data/raw/`.

### 4. Start the backend

```bash
cd app/backend
uvicorn main:app --reload --port 8000
```

API docs available at: `http://localhost:8000/docs`

### 5. Start the frontend

```bash
cd app/frontend
npm install
npm run dev
```

Open: `http://localhost:5173`

---

## 🧠 Model Architecture

```
Input [B, 3, 224, 224]
        ↓
EfficientNet-B0 Backbone    ← pretrained on ImageNet
(4,769,152 params)
        ↓
[B, 1280]
        ↓
Dropout(0.4)
        ↓
Linear(1280 → 512) + ReLU
        ↓
Dropout(0.2)
        ↓
Linear(512 → 38)
        ↓
Logits [B, 38]
```

**Total parameters:** 5,444,518

---

## 🏋️ Training

Training uses a two-phase strategy:

| | Phase 1 | Phase 2 |
|---|---|---|
| **Backbone** | Frozen | Unfrozen |
| **Learning rate** | 1e-3 | 1e-4 |
| **Optimizer** | AdamW | AdamW |
| **Scheduler** | CosineAnnealingLR | CosineAnnealingLR |
| **Epochs** | 14 (early stop) | 10 |
| **Val Accuracy** | 97.11% | **99.83%** |

To train the model:

```bash
# Phase 1
python src/trainer.py --phase 1 --epochs 15 --lr 1e-3

# Phase 2
python src/trainer.py --phase 2 --epochs 10 --lr 1e-4
```

---

## 📊 Results

| Model | Params | Epochs | Val Accuracy | F1-score | Val Loss |
|-------|--------|--------|-------------|---------|---------|
| PlantCNN (scratch) | 1.7M | 15 | 97.89% | 0.9788 | 0.0610 |
| EfficientNet Phase 1 | 675K | 14 | 97.11% | 0.9709 | 0.0863 |
| **EfficientNet Phase 1+2** | **5.4M** | **24** | **99.83%** | **0.9983** | **0.0055** |

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Single image prediction |
| `POST` | `/predict/batch` | Batch prediction |
| `GET` | `/health` | Service health check |
| `GET` | `/docs` | Swagger UI documentation |

**Example request:**

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@leaf_photo.jpg"
```

**Example response:**

```json
{
  "predictions": [
    {"class_name": "Strawberry___Leaf_scorch", "confidence": 0.9949, "rank": 1},
    {"class_name": "Strawberry___healthy",     "confidence": 0.0041, "rank": 2},
    {"class_name": "Raspberry___healthy",      "confidence": 0.0008, "rank": 3}
  ],
  "top1_class":   "Strawberry___Leaf_scorch",
  "top1_conf":    0.9949,
  "severity":     "high",
  "treatment":    "Apply appropriate fungicide or pesticide. Remove infected parts.",
  "inference_ms": 68.96
}
```

---

## 🌱 Supported Plants & Diseases

| Plant | Diseases |
|-------|---------|
| Apple | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| Corn | Cercospora Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| Grape | Black Rot, Esca, Leaf Blight, Healthy |
| Tomato | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria, Spider Mites, Target Spot, Mosaic Virus, TYLCV, Healthy |
| Potato | Early Blight, Late Blight, Healthy |
| Pepper | Bacterial Spot, Healthy |
| Peach | Bacterial Spot, Healthy |
| Cherry | Powdery Mildew, Healthy |
| Strawberry | Leaf Scorch, Healthy |
| + 5 more | Blueberry, Orange, Raspberry, Soybean, Squash |

---

## 📦 Requirements

```
torch>=2.0.0
torchvision>=0.15.0
fastapi>=0.110.0
uvicorn>=0.29.0
python-multipart>=0.0.9
Pillow>=10.0.0
numpy>=1.24.0
scikit-learn>=1.2.0
pydantic>=2.0.0
```

---

## 📁 Dataset

**New Plant Diseases Dataset** — Kaggle
- 87,867 images total
- 38 disease classes
- 14 plant species
- Train split: 70,295 images
- Val split: 17,572 images

Download: [kaggle.com/datasets/vipoooool/new-plant-diseases-dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

---

## 📄 License

This project is developed as a diploma project for IITU Kazakhstan.
