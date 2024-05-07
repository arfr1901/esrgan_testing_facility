import os
import tensorflow as tf
import numpy as np
from PIL import Image


# Utility for image loading and preprocessing
def load_img(path_to_img, scale_factor=4, save_path="downsampled_image.jpg"):
    img = tf.io.read_file(path_to_img)
    img = tf.io.decode_image(img, channels=3)
    img = img.numpy()
    hr = img = Image.fromarray(img)
    if not scale_factor:
      width, height = 512, 512
      scale_factor = 4
    else:
      width, height = img.size
    if save_path:
      lr = img = img.resize(
          (width // scale_factor, height // scale_factor),
          Image.BICUBIC)
      lr.save(save_path)
    img = np.asarray(img)
    img = tf.cast(img, dtype=tf.float32)
    img = img[tf.newaxis, :]
    return img


def create_low_high_res_pairs(file_names, hr_image_dir, output_dir, downscale_factor=4):
    os.makedirs(output_dir, exist_ok=True)

    for idx, image_name in enumerate(file_names, start=0):
        full_path = os.path.join(hr_image_dir, image_name)
        img = Image.open(full_path)

        # Calculate new dimensions that are divisible by downscale_factor
        new_width = (img.width // downscale_factor) * downscale_factor
        new_height = (img.height // downscale_factor) * downscale_factor

        # Resize to the new dimensions to ensure perfect downscaling
        img_resized = img.resize((new_width, new_height), Image.BICUBIC)

        # Save the resized high-resolution image
        ground_truth_name = f"ground_truth_{idx}.png"
        img_resized.save(os.path.join(output_dir, ground_truth_name))

        # Downscale to create the low-resolution version
        low_res = img_resized.resize((new_width // downscale_factor, new_height // downscale_factor), Image.BICUBIC)
        low_res_name = f"low_res_{idx}.png"
        low_res.save(os.path.join(output_dir, low_res_name))

