import onnxruntime as ort
import numpy as np

# Load the ONNX model
onnx_model_path = "baseline.onnx"
session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])

# Create a dummy input tensor (adjust shape based on your model)
input_tensor = np.random.rand(1, 1, 256, 65).astype(np.float32)

# Prepare input dictionary
input_name = session.get_inputs()[0].name
input_data = {input_name: input_tensor}

# Run inference
outputs = session.run(None, input_data)

# Print the model output
print("ONNX Model Output:", outputs)

