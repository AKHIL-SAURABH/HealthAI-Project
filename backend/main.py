from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image
import io
import json
from tensorflow.keras.models import load_model

# -----------------------------
# Paths
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent  # HealthAI-Project root
MODELS_DIR = BASE_DIR / "models"

# Pneumonia (simple) model
XRAY_SIMPLE_MODEL_PATH = MODELS_DIR / "xray_disease_model.h5"
XRAY_SIMPLE_CLASS_MAPPING_PATH = MODELS_DIR / "xray_class_mapping.json"

# CheXpert multi-disease model
CHEXPERT_MODEL_PATH = MODELS_DIR / "xray_chexpert_multidisease_model.h5"
CHEXPERT_LABELS_PATH = MODELS_DIR / "xray_chexpert_labels.json"

IMG_SIZE = (224, 224)  # both models trained with 224x224

# -----------------------------
# FastAPI app setup
# -----------------------------

app = FastAPI(title="HealthAI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Load models & mappings
# -----------------------------

print("Loading X-ray models and mappings...")

xray_simple_model = None
simple_idx_to_class: Dict[int, str] = {}

chexpert_model = None
chexpert_labels: List[str] = []

# Load simple pneumonia model (if available)
try:
    if XRAY_SIMPLE_MODEL_PATH.exists() and XRAY_SIMPLE_CLASS_MAPPING_PATH.exists():
        xray_simple_model = load_model(XRAY_SIMPLE_MODEL_PATH)

        with open(XRAY_SIMPLE_CLASS_MAPPING_PATH, "r") as f:
            raw_map = json.load(f)  # keys as strings
            simple_idx_to_class = {int(k): v for k, v in raw_map.items()}

        print("✅ Simple pneumonia model loaded.")
        print("Simple class mapping:", simple_idx_to_class)
    else:
        print("⚠️ Simple pneumonia model or mapping file not found.")

except Exception as e:
    print("❌ Error loading simple X-ray model:", e)
    xray_simple_model = None
    simple_idx_to_class = {}

# Load CheXpert multi-disease model (if available)
try:
    if CHEXPERT_MODEL_PATH.exists() and CHEXPERT_LABELS_PATH.exists():
        chexpert_model = load_model(CHEXPERT_MODEL_PATH)

        with open(CHEXPERT_LABELS_PATH, "r") as f:
            chexpert_labels = json.load(f)  # list of disease names in correct order

        print("✅ CheXpert multi-disease model loaded.")
        print("CheXpert labels:", chexpert_labels)
    else:
        print("⚠️ CheXpert model or labels file not found.")

except Exception as e:
    print("❌ Error loading CheXpert model:", e)
    chexpert_model = None
    chexpert_labels = []

# -----------------------------
# Helper function
# -----------------------------

def preprocess_xray_image(file_bytes: bytes) -> np.ndarray:
    """Convert uploaded image bytes into model-ready numpy array."""
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1, H, W, 3)
    return arr

# -----------------------------
# Routes
# -----------------------------

@app.get("/health")
def health_check():
    return {"status": "ok"}

# --------- SIMPLE: PNEUMONIA vs NORMAL ---------

@app.post("/predict-xray")
async def predict_xray(file: UploadFile = File(...)):
    """
    Simple model: Predict whether a chest X-ray indicates NORMAL or PNEUMONIA.
    """
    if xray_simple_model is None or not simple_idx_to_class:
        raise HTTPException(status_code=500, detail="Simple pneumonia model not loaded on server.")

    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    img_arr = preprocess_xray_image(file_bytes)

    # Model outputs a probability between 0 and 1
    prob = float(xray_simple_model.predict(img_arr)[0][0])

    # threshold 0.5
    pred_index = 1 if prob >= 0.5 else 0
    pred_label = simple_idx_to_class.get(pred_index, str(pred_index))

    return {
        "predicted_label": pred_label,
        "pneumonia_probability": prob
    }

# --------- ADVANCED: MULTI-DISEASE (CheXpert) ---------

@app.post("/predict-xray-multidisease")
async def predict_xray_multidisease(file: UploadFile = File(...)):
    """
    Multi-disease model (CheXpert): Predict probabilities for multiple chest conditions.
    """
    if chexpert_model is None or not chexpert_labels:
        raise HTTPException(status_code=500, detail="CheXpert multi-disease model not loaded on server.")

    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    img_arr = preprocess_xray_image(file_bytes)

    # Model outputs a vector of probabilities, one per label
    probs = chexpert_model.predict(img_arr)[0]  # shape: (len(chexpert_labels),)

    predictions = {
        chexpert_labels[i]: float(probs[i])
        for i in range(len(chexpert_labels))
    }

    # Also return top 3 conditions sorted by probability
    top3 = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:3]

    return {
        "predictions": predictions,
        "top3": [
            {"label": label, "probability": prob}
            for label, prob in top3
        ]
    }

# If you want to run directly with: python main.py
# Uncomment below:
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
