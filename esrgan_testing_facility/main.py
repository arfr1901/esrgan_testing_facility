import benchmark_interface as bm
from utility import methods
from utility import paths


def main():
    # controller.check_validation_dataset_dir()
    # controller.check_testing_dataset_dir()

    # bm.push_model_to_device("esrgan_drq.tflite", "192.168.8.120:43143")
    # bm.push_model_to_device("esrgan_fiq.tflite", "192.168.8.120:43143")
    # bm.push_model_to_device("esrgan_f16q.tflite", "192.168.8.120:43143")

    bm.benchmark(methods.quantization_strategies)
    for record in bm.records:
        print(record)


if __name__ == '__main__':
    main()

