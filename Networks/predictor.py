import numpy as np
import onnxruntime as ort


class MobilenetPosPredictor():
    def __init__(self, resolution_inp=256, resolution_op=256):
        self.resolution_inp = resolution_inp
        self.resolution_op = resolution_op
        self.MaxPos = resolution_inp * 1.1
        self.session = None

    def restore(self, model_path):
        # Use ONNX model if available, fall back to .h5 with TF
        onnx_path = model_path.replace('.h5', '.onnx')
        import os
        if os.path.exists(onnx_path):
            self.session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
            self.input_name = self.session.get_inputs()[0].name
        else:
            # Fallback to TensorFlow
            import tensorflow as tf
            from Networks import mobilenet_v2
            self.model = tf.keras.models.load_model(
                model_path,
                custom_objects={'relu6': mobilenet_v2.relu6},
                compile=False
            )
            self.session = None

    def predict(self, image):
        x = image[np.newaxis, :, :, :].astype(np.float32)
        if self.session is not None:
            pos = self.session.run(None, {self.input_name: x})[0]
        else:
            pos = self.model.predict(x=x, verbose=0)
        pos = np.squeeze(pos)
        return pos * self.MaxPos

    def predict_batch(self, images):
        raise NotImplementedError
