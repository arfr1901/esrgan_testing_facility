import tensorflow as tf
from utility import paths
import numpy as np
from PIL import Image
import os


# def representative_dataset_gen():
#     for path_to_img in paths.calibration_dataset:
#         img = tf.io.read_file(paths.train_hr_dir + path_to_img)
#         img = tf.io.decode_image(img, channels=3)
#         img = img.numpy()
#         img = Image.fromarray(img)
#
#         # width, height = img.size
#         img = img.resize(
#             (128, 128),
#             Image.BICUBIC)
#
#         img = np.asarray(img)
#         img = tf.cast(img, dtype=tf.float32)
#         img = img[tf.newaxis, :]
#         yield [img]


def representative_dataset_gen_test():
    # Ensure correct directory handling
    for path_to_img in paths.calibration_dataset_test:
        # Correct path concatenation
        full_path_to_img = os.path.join(paths.high_res_dataset_path, path_to_img)

        img = tf.io.read_file(full_path_to_img)
        img = tf.io.decode_image(img, channels=3, expand_animations=False)  # Ensure image is not animated
        img = img.numpy()
        img = Image.fromarray(img)

        # Resize the image as necessary
        img = img.resize((128, 128), Image.BICUBIC)

        img = np.asarray(img)
        img = tf.cast(img, dtype=tf.float32)
        img = img[tf.newaxis, :]  # Add batch dimension
        yield [img]