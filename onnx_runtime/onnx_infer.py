import onnxruntime as ort
import onnx
import numpy as np
import time

# Load your ONNX model
model_path = r"C:\Users\fenel\Documents\dcase2024_task1_baseline\onnx_runtime\baseline_dynamic.onnx"
load_model = onnx.load(model_path)
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

dummy_input = np.random.randn(1, 1, 256, 65).astype(np.float32)
outputs = session.run(None, {input_name: dummy_input})
print("Inference output:", outputs)
print("Opset version:", load_model.opset_import[0].version)

# Warm-up (to avoid first-run overhead)
for _ in range(10):
    session.run(None, {input_name: dummy_input})

# Measure inference time
N = 100
start = time.time()
for _ in range(N):
    outputs = session.run(None, {input_name: dummy_input})
end = time.time()

total_time = end - start
avg_time_ms = (total_time / N) * 1000

print(f"Total time for {N} inferences: {total_time:.4f} seconds")
print(f"Average inference time: {avg_time_ms:.2f} ms")