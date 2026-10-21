import os
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from enum import Enum
import tensorflow as tf
from PIL import Image
import io

# Import the prediction logger function
try:
    from log_predictions import log_model_predictions
except ImportError:
    def log_model_predictions(*args, **kwargs):
        pass

# Initialize FastAPI app
app = FastAPI(
    title="Casting Defect Classification API",
    description="REST API for serving casting defect classifier models with a model selection dropdown.",
    version="1.9"
)

# Define paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
MODEL_STORE_DIR = os.path.join(BASE_DIR, "model_store")

MODEL_PATHS = {
    "transfer_learning": os.path.join(MODEL_STORE_DIR, "casting_defect_transfer_model.keras"),
    "custom_cnn": os.path.join(MODEL_STORE_DIR, "casting_defect_classifier.keras")
}

loaded_models = {}

def load_model_cached(model_name: str):
    """Load model with caching."""
    if model_name in loaded_models:
        return loaded_models[model_name]
    
    model_path = MODEL_PATHS.get(model_name)
    if not model_path or not os.path.exists(model_path):
        return None
    
    try:
        print(f"[INFO] Loading model '{model_name}' from {model_path}")
        # Use TensorFlow directly with safe_mode disabled for Keras 3 compat
        model = tf.keras.models.load_model(model_path)
        loaded_models[model_name] = model
        print(f"[SUCCESS] Loaded model '{model_name}'")
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load model '{model_name}': {str(e)[:200]}")
        return None

print(f"[INIT] MODEL_STORE_DIR={MODEL_STORE_DIR}")
for name, path in MODEL_PATHS.items():
    exists = os.path.exists(path)
    print(f"[INIT] {name}: {path} (exists={exists})")

class ModelChoice(str, Enum):
    transfer_learning = "transfer_learning"
    custom_cnn = "custom_cnn"

IMG_SIZE = (224, 224)

class InferenceResponse(BaseModel):
    model_used: str
    prediction: float
    result: str
    confidence: float
    
    class Config:
        protected_namespaces = ()

@app.get("/")
def health_check():
    return {
        "status": "healthy", 
        "available_models": list(MODEL_PATHS.keys())
    }

@app.post("/predict", response_model=InferenceResponse)
async def predict_defect(
    file: UploadFile = File(...), 
    model_name: ModelChoice = Form(ModelChoice.transfer_learning, description="Select model architecture from dropdown")
):
    selected_model_key = model_name.value
    print(f"[REQUEST] Prediction request for model '{selected_model_key}'")
    
    if selected_model_key not in MODEL_PATHS:
        raise HTTPException(
            status_code=400, 
            detail=f"Model '{selected_model_key}' not recognized. Choose from: {list(MODEL_PATHS.keys())}"
        )
    
    print(f"[LOAD] Loading model '{selected_model_key}'...")
    model = load_model_cached(selected_model_key)
    if model is None:
        print(f"[ERROR] Could not load model '{selected_model_key}'")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to load model '{selected_model_key}'. Check server logs."
        )
    
    print(f"[PROCESS] Processing image from '{file.filename}'...")
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize(IMG_SIZE, Image.Resampling.BILINEAR)
        
        img_array = np.array(image, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        
        print(f"[INFER] Running inference with {selected_model_key}...")
        prediction = model.predict(img_array, verbose=0)
        score = float(prediction[0][0])
        print(f"[RESULT] Prediction score: {score}")
        
        result_label = "ok" if score >= 0.50 else "defective"
        confidence_score = abs(score - 0.50) * 2 * 100
        
        # Log prediction
        input_df = pd.DataFrame({"filename": [file.filename]})
        preds = np.array([score])
        log_model_predictions(
            input_data=input_df,
            predictions=preds,
            model_version=selected_model_key,
            score=score,
            predicted_label=result_label
        )
        
        print(f"[RESPONSE] {result_label} (confidence: {confidence_score:.1f}%)")
        return InferenceResponse(
            model_used=selected_model_key,
            prediction=score,
            result=result_label,
            confidence=confidence_score
        )
        
    except HTTPException as he:
        print(f"[ERROR] HTTP Exception: {he.detail}")
        raise he
    except Exception as e:
        print(f"[ERROR] Exception during inference: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
