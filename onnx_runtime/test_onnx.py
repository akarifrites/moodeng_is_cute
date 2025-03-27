import onnx

model_path = r"C:\Users\fenel\Documents\dcase2024_task1_baseline\onnx_runtime\baseline.onnx"

try:
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    print("✅ The ONNX model is valid!")
except onnx.onnx_cpp2py_export.checker.ValidationError as e:
    print(f"❌ ONNX Validation Error: {e}")
except Exception as e:
    print(f"❌ Error loading ONNX model: {e}")