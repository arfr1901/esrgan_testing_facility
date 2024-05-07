import tensorflow as tf
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import os
from skimage import io
from PIL import Image


# function used to facilitate both validation and testing
def evaluate(model, dataset_dir):
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Initialize lists to store evaluation results
    psnr_scores = []
    ssim_scores = []

    # Assume an equal number of low-res and ground-truth results
    num_images = len([name for name in os.listdir(dataset_dir) if "low_res" in name])

    for i in range(2):
        low_res_path = os.path.join(dataset_dir, f"low_res_{i}.png")
        high_res_path = os.path.join(dataset_dir, f"ground_truth_{i}.png")

        # Load results
        low_res_image = io.imread(low_res_path)
        high_res_image = io.imread(high_res_path)

        # Resize the low-resolution image to match the input size expected by the model
        low_res_image = Image.fromarray(low_res_image)
        low_res_image = low_res_image.resize((128, 128), Image.BICUBIC)  # Adjust size as needed
        low_res_image = np.array(low_res_image, dtype=np.float32)
        low_res_image = np.expand_dims(low_res_image, axis=0)  # Add batch dimension

        # Ensure the target image is also correctly shaped
        high_res_image = high_res_image.astype(np.float32)

        # Perform inference on the low-resolution image
        interpreter.set_tensor(input_details[0]['index'], low_res_image)
        interpreter.invoke()
        output_image = interpreter.tensor(output_details[0]['index'])()

        # Ensure output image is compatible for comparison
        output_image = np.clip(output_image.squeeze(), 0, 255).astype(np.uint8)

        # Calculate PSNR and SSIM between the output image and the high-resolution image
        psnr = peak_signal_noise_ratio(high_res_image, output_image, data_range=255)

        # print("High-res image size:", high_res_image.shape)
        # print("Output image size:", output_image.shape)

        ssim = structural_similarity(high_res_image, output_image, win_size=3, data_range=255, multichannel=True)

        # Append scores to the lists
        psnr_scores.append(psnr)
        ssim_scores.append(ssim)

    # Calculate average PSNR and SSIM scores
    avg_psnr = np.mean(psnr_scores)
    avg_ssim = np.mean(ssim_scores)

    return avg_psnr, avg_ssim

