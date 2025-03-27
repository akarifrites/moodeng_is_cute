import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process

model_fp32 = r"C:\Users\fenel\Documents\dcase2024_task1_baseline\onnx_runtime\baseline.onnx"

model_fp32_optimized = r'c:\Users\fenel\Documents\dcase2024_task1_baseline\onnx_runtime\baseline_optimized.onnx'

model_quantized = r'c:\Users\fenel\Documents\dcase2024_task1_baseline\onnx_runtime\baseline_dynamic.onnx'

# Run preprocessing optimization
quant_pre_process(model_fp32, model_fp32_optimized)


quantize_dynamic(model_input=model_fp32_optimized,
                model_output=model_quantized,
                weight_type=QuantType.QInt8
)