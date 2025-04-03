from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process
import numpy as np
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader, Subset
from dataset.dcase24 import get_training_set

class MelSpec(nn.Module):
    def __init__(self):
        super(MelSpec, self).__init__()
        resample = torchaudio.transforms.Resample(
            orig_freq=44100, 
            new_freq=32000
        )

        # define mel spectrogram
        mel = torchaudio.transforms.MelSpectrogram(sample_rate=32000, 
                                                    n_fft=4096, 
                                                    win_length=3072,
                                                    hop_length=500, 
                                                    n_mels=256)
        
        self.mel = torch.nn.Sequential(
            resample,
            mel
        )

    def forward(self,x):
        # print("Input tensor shape in model.forward:", x.shape)
        x = self.mel(x)
        x = (x + 1e-5).log()
        return x

class CalibrationData(CalibrationDataReader):
    def __init__(self, calibration_loader, MelSpecGenerator):
        self.loader_iter = iter(calibration_loader)
        self.mel_gen = MelSpecGenerator

    def get_next(self):
        try:
            inputs = next(self.loader_iter)
            raw_waveform = inputs[0]
            mel_spec = self.mel_gen(raw_waveform)

            # Ensure mel_spec is on CPU and converted to numpy
            if isinstance(mel_spec, torch.Tensor):
                mel_spec = mel_spec.cpu().numpy()

            return {'input': mel_spec}
        except StopIteration:
            return None

# Get calibration data subset
train_dataset = get_training_set()
num_calibration_samples = int(0.05 * len(train_dataset))
calibration_indices = np.random.choice(len(train_dataset), num_calibration_samples, replace=False)
calibration_data = Subset(train_dataset, calibration_indices)

# Calibration DataLoader
calibration_loader = DataLoader(dataset=calibration_data,
                                batch_size=256,
                                num_workers=0,
                                shuffle=False)

model_fp32 = r"C:\Users\fenel\Documents\dcase2024_task1_baseline\onnx_runtime\baseline.onnx"

model_fp32_optimized = r'c:\Users\fenel\Documents\dcase2024_task1_baseline\onnx_runtime\baseline_optimized.onnx'

model_quantized = r'c:\Users\fenel\Documents\dcase2024_task1_baseline\onnx_runtime\baseline_static.onnx'



# Run preprocessing optimization
quant_pre_process(model_fp32, model_fp32_optimized)

# Instantiate calibration data reader
data_reader = CalibrationData(calibration_loader, MelSpec())
quantize_static(model_input=model_fp32_optimized,
                model_output='baseline_static.onnx',
                calibration_data_reader=data_reader,
                quant_format='QDQ', 
                per_channel=True, 
                activation_type=QuantType.QUInt8,
                weight_type=QuantType.QInt8,
                optimize_model=False,  # Important for older runtime support
                use_external_data_format=False
                )