import os
import datetime as dt
import pandas as pd
import numpy as np
import joblib

# Define paths relative to project structure
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
FEATURE_STORE_DIR = os.path.join(BASE_DIR, "feature_store")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Ensure necessary directories exist
os.makedirs(FEATURE_STORE_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

def log_model_predictions(
    input_data: pd.DataFrame, 
    predictions: np.ndarray, 
    model_version: str = "v1.0", 
    score: float = None, 
    predicted_label: str = None, 
    **kwargs
) -> str:
    """
    Logs incoming features and their corresponding model predictions 
    to a persistent CSV file for tracking, auditing, and drift detection.
    """
    log_file_path = os.path.join(LOGS_DIR, "prediction_logs.csv")
    
    # Create a copy of the input dataframe to avoid mutating original data
    log_df = input_data.copy()
    
    # Append metadata and prediction info safely
    log_df["score"] = score
    log_df["predicted_label"] = predicted_label
    log_df["prediction"] = predictions
    log_df["model_version"] = model_version
    log_df["logged_at"] = dt.datetime.utcnow().isoformat()
    
    # Append to CSV log store (create if it doesn't exist, append if it does)
    if os.path.exists(log_file_path):
        log_df.to_csv(log_file_path, mode='a', header=False, index=False)
    else:
        log_df.to_csv(log_file_path, mode='w', header=True, index=False)
        
    print(f"Successfully logged {len(predictions)} prediction records to {log_file_path}")
    return log_file_path

if __name__ == "__main__":
    print("Initializing prediction logger test with dummy data...")
    
    # Sample test DataFrame matching your pipeline context
    sample_input = pd.DataFrame({
        "sensor_pressure": [101.3, 102.5],
        "batch_id": ["batch_A", "batch_B"]
    })
    
    # Mock model predictions
    sample_preds = np.array([0, 1])
    
    # Execute logger test
    log_model_predictions(
        sample_input, 
        sample_preds, 
        model_version="v1.0-test", 
        score=0.95, 
        predicted_label="defective"
    )
    print("Prediction logging execution complete.")