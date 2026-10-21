import os
import tensorflow as tf

# Define paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = CURRENT_DIR # Adjust if your script sits inside a subfolder
DATA_DIR = os.path.join(BASE_DIR, "data", "casting_data", "casting_data")
TEST_DIR = os.path.join(DATA_DIR, "test")
MODEL_STORE_DIR = os.path.join(BASE_DIR, "model_store")

# Configuration (Must match what you trained them on)
IMG_SIZE = (224, 224) 
BATCH_SIZE = 16

def evaluate_models():
    print("Loading test dataset...")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR, 
        image_size=IMG_SIZE, 
        batch_size=BATCH_SIZE, 
        shuffle=False
    )
    
    # Paths to your saved models
    cnn_path = os.path.join(MODEL_STORE_DIR, "casting_defect_classifier.keras")
    transfer_path = os.path.join(MODEL_STORE_DIR, "casting_defect_transfer_model.keras")
    
    models_to_test = {}
    
    if os.path.exists(cnn_path):
        models_to_test["Custom CNN"] = cnn_path
    else:
        print(f"Warning: Custom CNN model not found at {cnn_path}")
        
    if os.path.exists(transfer_path):
        models_to_test["Transfer Learning (MobileNetV2)"] = transfer_path
    else:
        print(f"Warning: Transfer Learning model not found at {transfer_path}")
        
    if not models_to_test:
        print("No models found to evaluate! Train them first.")
        return

    print("\n" + "="*50)
    print(f"{'Model Name':<35} | {'Test Loss':<10} | {'Test Accuracy':<10}")
    print("="*50)

    for name, path in models_to_test.items():
        # Load the saved model file
        model = tf.keras.models.load_model(path)
        
        # Evaluate on the test dataset
        loss, accuracy = model.evaluate(test_ds, verbose=0)
        
        print(f"{name:<35} | {loss:.4f}     | {accuracy*100:.2f}%")

    print("="*50)

if __name__ == "__main__":
    evaluate_models()