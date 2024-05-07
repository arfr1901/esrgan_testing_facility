import tensorflow as tf
import kagglehub
import os
from pathlib import Path

saved_model = kagglehub.model_download("kaggle/esrgan-tf2/tensorFlow2/esrgan-tf2")
print("Path to model files:", saved_model)

local_og_model_path = "C:/Users/darks/.cache/kagglehub/models/kaggle/esrgan-tf2/tensorFlow2/esrgan-tf2/1"
relative_og_model_path = "../original_model/"

saved_model_path = Path(saved_model).absolute()

# Load your pre-trained ESRGAN model
uncompressed_model = tf.saved_model.load(saved_model_path)

local_train_hr_dir = 'C:/a_Skola/ImageSuperResolution/python/datasets/DIV2K_train_HR/'
relative_train_hr_dir = '../datasets/DIV2K_train_HR/'

# For representative dataset generator function
# calibration_dataset = os.listdir(relative_train_hr_dir)[:100]

validation_dataset = '../esrgan_testing_facility/datasets/validation_dataset/'

testing_dataset = '../esrgan_testing_facility/datasets/testing_dataset/'

# dataset with 512x512 results
high_res_dataset_path = '../esrgan_testing_facility/datasets/512x512/'

cwd = os.getcwd()

what = os.path.abspath(high_res_dataset_path)

calibration_dataset_test = sorted(os.listdir(high_res_dataset_path))[99:100]
validation_dataset_file_names = sorted(os.listdir(high_res_dataset_path))[101:102]
testing_dataset_file_names = sorted(os.listdir(high_res_dataset_path))[103:104]

