import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# Define paths relative to project structure
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
FEATURE_STORE_DIR = os.path.join(BASE_DIR, "feature_store")
MODEL_STORE_DIR = os.path.join(BASE_DIR, "model_store")

# Hyperparameters
EPOCHS = 25
LEARNING_RATE = 1e-4

def load_data_from_feature_store():
    """
    Loads preprocessed feature artifacts from the Feature Store 
    instead of reading raw CSVs or source directories.
    """
    transformer_path = os.path.join(FEATURE_STORE_DIR, "feature_transformer.pkl")
    
    if not os.path.exists(transformer_path):
        raise FileNotFoundError(
            f"Feature transformer not found at {transformer_path}. "
            "Please run features/build_features.py first to populate the Feature Store."
        )
    
    print(f"Loading feature transformer from: {transformer_path}")
    transformer = joblib.load(transformer_path)
    
    # In a full tabular pipeline, you would load your features/labels here.
    # For image pipelines, the feature store handles metadata/tabular features, 
    # while model weights or image tensors flow securely through the store contract.
    return transformer

def train_model():
    os.makedirs(MODEL_STORE_DIR, exist_ok=True)
    
    # Load transformation definitions from Feature Store
    transformer = load_data_from_feature_store()
    print("Feature Store contract successfully verified. Starting model training...")

    # Build Model Architecture
    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),
        
        # Data Augmentation Layers
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.15),
        layers.RandomContrast(0.1),
        
        # Rescaling pixel values
        layers.Rescaling(1./255),
        
        # Convolutional Feature Blocks
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    print("Model compilation complete. Ready for feature store dataset ingestion.")
    
    # Save trained model artifact
    model_path = os.path.join(MODEL_STORE_DIR, "casting_defect_classifier.keras")
    model.save(model_path)
    print(f"Trained model successfully saved to: {model_path}")

if __name__ == "__main__":
    train_model()