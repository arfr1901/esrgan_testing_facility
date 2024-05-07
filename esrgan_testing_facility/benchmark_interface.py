from utility import metrics, methods, devices as dev
from tflitebenchmark import TFLiteBenchmark
import tflitebenchmark as tflbm


def push_model_to_device(model_path, device_id):
    """Push the optimized model to the device only if it has a new optimization strategy"""
    tflbm.run_command(["adb", "-s", device_id, "push", model_path, "/data/local/tmp"])


def avg_inference_time(device_id, model_name, use_gpu=False):
    inf_benchmark = TFLiteBenchmark(
        device_id=device_id,
        model_name=model_name,
        num_runs=1,
        num_threads=4
    )

    push_model_to_device(methods.model_path_of(model_name), device_id)
    tflbm.clear_adb_log(device_id)
    inf_benchmark.start_inference_time_benchmark(model_name, use_gpu)
    return inf_benchmark.get_inference_time_results_ms()


def file_size(path):
    return metrics.get_file_size_in_mb_for(methods.model_path_of(path))


def compression_ratio(sz):
    return metrics.get_compression_ratio_for(sz)


# List to hold all the records
records = []

default_model = "ESRGAN.tflite"


def benchmark(opt_strategies):
    # Iterate over each device and optimization strategy combination
    for device in dev.devices:
        for strategy in opt_strategies:
            compressed_model = f"esrgan_{strategy}.tflite"
            if strategy == "drq":
                compressed_model = "esrgan_drq.tflite"
            if strategy == "fiq":
                compressed_model = "esrgan_fiq.tflite"
            if strategy == "f16q":
                compressed_model = "esrgan_f16q.tflite"

            # Get the metrics for the model
            cpu_inference_time = avg_inference_time(device["id"], compressed_model)
            # gpu_inference_time = avg_inference_time(device["id"], compressed_model, True)

            # Add the metrics to the records list
            records.append({
                "Device": device["name"],
                "Quantization Technique": strategy,
                "CPU time (ms)": cpu_inference_time,
                # "GPU time (ms)": gpu_inference_time
            })
