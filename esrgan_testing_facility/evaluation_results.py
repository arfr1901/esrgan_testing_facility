import evaluate as ev
from utility import paths
from esrgan import EsrganNeuralNetwork

esrgan_fiq = EsrganNeuralNetwork()
esrgan_drq = EsrganNeuralNetwork()
esrgan_f16q = EsrganNeuralNetwork()

esrgan_fiq.quantize_using("fiq")
esrgan_drq.quantize_using("drq")
esrgan_f16q.quantize_using("f16q")

fiq_val_avg_psnr, fiq_val_avg_ssim = ev.evaluate(esrgan_fiq.tflite_model, paths.validation_dataset)
fiq_test_avg_psnr, fiq_test_avg_ssim = ev.evaluate(esrgan_fiq.tflite_model, paths.testing_dataset)
print("Full Integer Quantization benchmarking results:")
print(f'Average Validation PSNR score: {fiq_val_avg_psnr}, Average Validation SSIM score: {fiq_val_avg_ssim}')
print(f'Average Testing PSNR score: {fiq_test_avg_psnr}, Average Testing SSIM score: {fiq_test_avg_ssim}')

drq_val_avg_psnr, drq_val_avg_ssim = ev.evaluate(esrgan_drq.tflite_model, paths.validation_dataset)
drq_test_avg_psnr, drq_test_avg_ssim = ev.evaluate(esrgan_drq.tflite_model, paths.testing_dataset)
print("Dynamic Range Quantization benchmarking results:")
print(f'Average Validation PSNR score: {drq_val_avg_psnr}, Average Validation SSIM score: {drq_val_avg_ssim}')
print(f'Average Testing PSNR score: {drq_test_avg_psnr}, Average Testing SSIM score: {drq_test_avg_ssim}')

f16q_val_avg_psnr, f16q_val_avg_ssim = ev.evaluate(esrgan_f16q.tflite_model, paths.validation_dataset)
f16q_test_avg_psnr, f16q_test_avg_ssim = ev.evaluate(esrgan_f16q.tflite_model, paths.testing_dataset)
print("Float 16 Quantization benchmarking results:")
print(f'Average Validation PSNR score: {f16q_val_avg_psnr}, Average Validation SSIM score: {f16q_val_avg_ssim}')
print(f'Average Testing PSNR score: {f16q_test_avg_psnr}, Average Testing SSIM score: {f16q_test_avg_ssim}')
