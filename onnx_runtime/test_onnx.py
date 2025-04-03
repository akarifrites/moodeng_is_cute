import onnx

model_path = r"C:\Users\fenel\Documents\dcase2024_task1_baseline\onnx_runtime\baseline_static.onnx"
model = onnx.load(model_path)

# Print all imported opsets
print("Original opsets:")
for opset in model.opset_import:
    print(f"Domain: '{opset.domain}', Version: {opset.version}")

# Modify in-place: downgrade ai.onnx.ml to version 3
for opset in model.opset_import:
    if opset.domain == "ai.onnx.ml" and opset.version > 3:
        print("Downgrading 'ai.onnx.ml' opset to 3")
        opset.version = 3

# Save the updated model
onnx.save(model, "onnx_runtime/baseline_static_fixed.onnx")
print("Patched model saved as 'baseline_static_fixed.onnx'")