import os
from utility import methods


def get_file_size_in_mb_for(file_name):

    # Get the size of the file in bytes
    file_size = os.path.getsize(file_name)

    # Convert bytes to kilobytes (1 KB = 1024 bytes)
    file_size_kb = file_size / 1024

    # Convert bytes to megabytes (1 MB = 1024 KB)
    file_size_mb = file_size_kb / 1024

    print(f"The size of the file is: {file_size} bytes")

    return file_size_mb


def complete_model_size(model_path, variables_path):
    try:
        # Get the size of the main model file and variables file
        model_size = os.path.getsize(model_path)
        variables_size = os.path.getsize(variables_path)

        # Convert bytes to megabytes (1 MB = 1,048,576 bytes)
        total_size_mb = (model_size + variables_size) / 1048576
        return total_size_mb
    except FileNotFoundError:
        print("One or more files not found.")
        return None


# Paths defined as constants for clarity and reusability
MODEL_PATH = 'C:/Users/darks/.cache/kagglehub/models/kaggle/esrgan-tf2/tensorFlow2/esrgan-tf2/1/saved_model.pb'
VARIABLES_PATH = 'C:/Users/darks/.cache/kagglehub/models/kaggle/esrgan-tf2/tensorFlow2/esrgan-tf2/1/variables/variables.data-00000-of-00001'


def get_compression_ratio_for(compressed_file):
    try:
        compressed_sz = os.path.getsize(compressed_file) / 1048576
        total_uncompressed_size = complete_model_size(MODEL_PATH, VARIABLES_PATH)
        if total_uncompressed_size is not None and total_uncompressed_size > 0:
            compression_ratio = compressed_sz / total_uncompressed_size
            percentage = (1 - compression_ratio) * 100
            return percentage
        else:
            return None
    except FileNotFoundError:
        print(f"File not found: {compressed_file}")
        return None
