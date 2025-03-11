import torch
checkpoint = torch.load("quantized_model.pt", map_location=torch.device("cpu"))
print(type(checkpoint))