import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from Networks import mobilenet_v2


class MobilenetPosPredictor():
    def __init__(self, resolution_inp=256, resolution_op=256):
        self.resolution_inp = resolution_inp
        self.resolution_op = resolution_op
        self.MaxPos = resolution_inp * 1.1
        self.model = None

        # Configure GPU memory growth for TF2
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"GPU config error: {e}")

    def restore(self, model_path):
        try:
            # Try TF2/Keras 2 style loading
            from tensorflow import keras
            self.model = keras.models.load_model(
                model_path,
                custom_objects={'relu6': mobilenet_v2.relu6},
                compile=False
            )
        except Exception:
            # Fallback for Keras 3
            self.model = tf.keras.models.load_model(
                model_path,
                custom_objects={'relu6': mobilenet_v2.relu6},
                compile=False
            )

    def predict(self, image):
        x = image[np.newaxis, :, :, :]
        pos = self.model.predict(x=x, verbose=0)
        pos = np.squeeze(pos)
        return pos * self.MaxPos

    def predict_batch(self, images):
        raise NotImplementedError
