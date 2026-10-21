import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Define paths relative to project structure
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
FEATURE_STORE_DIR = os.path.join(BASE_DIR, "feature_store")

# Ensure feature store directory exists
os.makedirs(FEATURE_STORE_DIR, exist_ok=True)

def build_feature_transformer():
    """
    Builds a pure, reproducible ColumnTransformer for feature engineering.
    Shared across training and serving to guarantee ZERO train/serving skew.
    """
    # Define feature schemas (modify columns according to your tabular/metadata schema if applicable)
    numeric_features = []       # e.g., casting temperature, pressure sensors
    categorical_features = []   # e.g., mold alloy batch, machine ID

    # Pure transformations for numeric features
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # Pure transformations for categorical features (silently ignores unknown values in serving)
    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combined ColumnTransformer pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    return preprocessor

def process_and_persist_features(df: pd.DataFrame, is_training: bool = True) -> np.ndarray:
    """
    Pure function: Same input always produces the same output.
    Fits the transformer during training and uses .transform() during serving to prevent skew.
    Persists the fitted transformer artifact to the Feature Store.
    """
    transformer_path = os.path.join(FEATURE_STORE_DIR, "feature_transformer.pkl")
    
    if is_training:
        transformer = build_feature_transformer()
        transformed_data = transformer.fit_transform(df) if not df.empty else np.array([])
        # Persist to Feature Store
        joblib.dump(transformer, transformer_path)
        print(f"Feature transformer fitted and persisted to: {transformer_path}")
    else:
        # Load existing transformer during serving to ensure zero skew
        if os.path.exists(transformer_path):
            transformer = joblib.load(transformer_path)
            transformed_data = transformer.transform(df) if not df.empty else np.array([])
            print(f"Loaded feature transformer from Feature Store for serving.")
        else:
            raise FileNotFoundError(f"Feature transformer not found at {transformer_path}. Run training feature pipeline first.")
            
    return transformed_data

if __name__ == "__main__":
    print("Initializing feature engineering test with dummy dataset...")
    sample_df = pd.DataFrame() # Replace with sample dataframe for testing
    processed = process_and_persist_features(sample_df, is_training=True)
    print("Feature pipeline execution complete.")