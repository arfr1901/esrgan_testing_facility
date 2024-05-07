import os
from utility import paths, preprocessor as pp
from PIL import Image


def check_validation_dataset_dir():
    if not os.listdir(paths.validation_dataset):
        pp.create_low_high_res_pairs(
            paths.validation_dataset_file_names,
            paths.high_res_dataset,  # Pass the directory path here
            paths.validation_dataset,
            downscale_factor=4
        )


def check_testing_dataset_dir():
    if not os.listdir(paths.testing_dataset):
        pp.create_low_high_res_pairs(
            paths.testing_dataset_file_names,
            paths.high_res_dataset,  # Pass the directory path here
            paths.testing_dataset,
            downscale_factor=4
        )


def check_image_dimensions(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.png'):  # Assuming the results are in PNG format
            img_path = os.path.join(directory, filename)
            with Image.open(img_path) as img:
                width, height = img.size
                if width % 4 != 0 or height % 4 != 0:
                    print(f"Image {filename} has dimensions {width}x{height}, which are not multiples of 4.")
                    return False
    print("All results have dimensions that are multiples of 4.")
    return True