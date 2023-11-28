import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the Keras model
keras_model = load_model("./models/bees-wasps.h5")

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
tflite_model = converter.convert()

# Save the converted model to a file
with open("_model.tflite", "wb") as f:
    f.write(tflite_model)