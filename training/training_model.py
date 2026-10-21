import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING logs
import tensorflow as tf
from tensorflow.keras import layers, models

# Path mappings
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data", "casting_data", "casting_data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
MODEL_STORE_DIR = os.path.join(BASE_DIR, "model_store")

os.makedirs(MODEL_STORE_DIR, exist_ok=True)

# Configuration
IMG_SIZE = (224, 224) 
BATCH_SIZE = 16
EPOCHS = 5

def train_and_evaluate():
    print(f"Loading training dataset from: {TRAIN_DIR}")
    if not os.path.exists(TRAIN_DIR):
        print(f"Error: Train directory not found at {TRAIN_DIR}.")
        return

    # Load datasets
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR, 
        image_size=IMG_SIZE, 
        batch_size=BATCH_SIZE, 
        shuffle=True
    )
    
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR, 
        image_size=IMG_SIZE, 
        batch_size=BATCH_SIZE, 
        shuffle=False
    )
    
    # Optimize pipeline performance for CPU
    train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.cache().prefetch(tf.data.AUTOTUNE)

    # 1. Build or Load MobileNetV2 Transfer Learning Model for Retraining
    base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                                   include_top=False,
                                                   weights='imagenet')
    base_model.trainable = False  # Freeze base layers initially

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs, outputs)

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    print("\nStarting model retraining...")
    model.fit(train_ds, epochs=EPOCHS)

    # Save the retrained model
    save_path = os.path.join(MODEL_STORE_DIR, "casting_defect_classifier.keras")
    model.save(save_path)
    print(f"Retrained model saved to: {save_path}")

    # Evaluate
    loss, accuracy = model.evaluate(test_ds, verbose=0)
    print(f"\nRetrained Model Test Loss: {loss:.4f} | Test Accuracy: {accuracy*100:.2f}%")

if __name__ == "__main__":
    train_and_evaluate()