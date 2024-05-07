import tensorflow as tf
from utility import paths, preprocessor as pp
import numpy as np
import calibrate as calib
from sklearn.cluster import KMeans
import tensorflow_hub as hub
model = hub.load("https://kaggle.com/models/kaggle/esrgan-tf2/frameworks/TensorFlow2/variations/esrgan-tf2/versions/1")


class EsrganNeuralNetwork():
    def __init__(self):
        self._tflite_model = None

    @property
    def tflite_model(self):
        return self._tflite_model

    # @param ["drq", "f16q", "fiq"]
    def quantize_using(self, quantization_strategy):
        model = paths.uncompressed_model
        concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

        concrete_func.inputs[0].set_shape([1, 512 // 4, 512 // 4, 3])

        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]    # drq

        if quantization_strategy == "f16q":   # float16
            converter.target_spec.supported_types = [tf.float16]
        elif quantization_strategy == "fiq":  # int8
            converter.representative_dataset = calib.representative_dataset_gen_test

        # 1 image = 17 s, 100 results => 28-30 mins
        tflite_model_data = converter.convert()
        model_path = f'results/esrgan_{quantization_strategy}.tflite'

        # store compressed model data in the specified directory
        with tf.io.gfile.GFile(model_path, 'wb') as f:
            f.write(tflite_model_data)

        self._tflite_model = model_path

    def prune_weights(self, num_clusters):
        # Gather all weights
        all_weights = []
        for layer in self.layers:
            if layer.type == 'dense':  # Assuming you have a list of layers
                all_weights.extend(layer.weights.flatten())

        # Convert to numpy array
        all_weights = np.array(all_weights)

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(all_weights.reshape(-1, 1))

        # Get centroids
        centroids = kmeans.cluster_centers_.squeeze()

        # Replace weights with nearest centroid
        for layer in self.layers:
            if layer.type == 'dense':
                flattened_weights = layer.weights.flatten()
                for i, weight in enumerate(flattened_weights):
                    closest_centroid = centroids[np.argmin(np.abs(centroids - weight))]
                    flattened_weights[i] = closest_centroid
                # Reshape back to original shape
                layer.weights = flattened_weights.reshape(layer.weights.shape)
