import subprocess
import re
import time


# Function to execute shell commands safely
def run_command(command, shell=False):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, shell=shell)
    return result.stdout.strip()


def clear_adb_log(device_id):
    """
    Clears the ADB log buffer for the given device.

    Args:
    device_id (str): The ID of the Android device.
    """
    command = ["adb", "-s", device_id, "logcat", "-c"]
    try:
        subprocess.run(command, check=True)
        print(f"ADB log buffer cleared for device {device_id}.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to clear ADB log buffer for device {device_id}: {e}")


# Class for benchmarking average inference time
# using Tensorflow's benchmarking app
class TFLiteBenchmark:
    def __init__(self, device_id, model_name, num_runs, num_threads):
        self.device_id = device_id
        self.model = model_name
        self.num_runs = num_runs
        self.num_threads = num_threads

    """Start the benchmarking process"""

    # Benchmarks the inference time of the model in relation
    # to the hardware of the device
    def start_inference_time_benchmark(self, model_name, use_gpu=False):
        inner_args = f"--graph=/data/local/tmp/{model_name} --num_threads={self.num_threads} --num_runs={self.num_runs}"
        if use_gpu:
            inner_args += " --use_gpu=true"

        args = f'\\\"{inner_args}\\\"'
        cmd = f"adb -s {self.device_id} shell am start -S -n org.tensorflow.lite.benchmark/.BenchmarkModelActivity --es args {args}"
        print("Running command:", cmd)
        run_command(cmd, shell=True)

    def get_inference_time_results_ms(self):
        pattern = re.compile(r"Inference \(avg\)\s*:\s*([\d.]+(?:e[+-]?\d+)?)")
        # pattern = re.compile(r"Inference \(avg\)\s*:\s*([\d.]+e[+-]?\d+)")
        matches = None
        timeout = time.time() + 60 * 10  # Set a timeout (e.g., 5 minutes)

        while not matches:
            if time.time() > timeout:
                raise TimeoutError("Timeout waiting for inference results")
            # Fetch logcat output
            logcat_output = run_command(
                ["adb", "-s", self.device_id, "logcat", "-d", "|", "findstr", "Inference timings"])
            # Clear logcat to avoid processing old messages again
            run_command(["adb", "-s", self.device_id, "logcat", "-c"])
            # Find matches
            matches = pattern.findall(logcat_output)
            if not matches:
                time.sleep(1)  # Wait a second before retrying to prevent high CPU usage

        avg_inference_time_us = float(matches[0])
        avg_inference_time_ms = avg_inference_time_us / 1000
        return avg_inference_time_ms
