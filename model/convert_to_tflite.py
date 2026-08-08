"""
Run this ONCE locally (not on Render) where you have your original
model/food_model.keras file. It produces model/food_model.tflite,
which uses far less memory at inference time.

Usage:
    pip install tensorflow
    python convert_to_tflite.py
"""

import tensorflow as tf

MODEL_PATH = "model/food_model.keras"
OUTPUT_PATH = "model/food_model.tflite"

print("Loading Keras model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optional: this quantization step shrinks the model further and
# speeds up inference, at a very small accuracy cost. Safe to keep.
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open(OUTPUT_PATH, "wb") as f:
    f.write(tflite_model)

print(f"Done. Saved to {OUTPUT_PATH}")
print(f"Original size vs new size — check with: ls -lh model/")
