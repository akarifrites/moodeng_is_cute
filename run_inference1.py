import torch
import torch.nn.functional as F
import torchaudio
import onnxruntime as ort
import sounddevice as sd
import numpy as np

# Load the ONNX model
onnx_model_path = "baseline.onnx"
session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])

# # Create a dummy input tensor (adjust shape based on your model)
# input_tensor = np.random.rand(1, 1, 256, 65).astype(np.float32)

# Prepare input dictionary
input_name = session.get_inputs()[0].name
# input_data = {input_name: input_tensor}

# # Run inference
# outputs = session.run(None, input_data)

# # Print the model output
# print("ONNX Model Output:", outputs)

# 🔹 Set sample rate (match incoming audio)
SAMPLE_RATE = 32000
DURATION = 2  # Length of each recording in seconds

# # 🔹 Mel Spectrogram transformation
# mel_transform = torchaudio.transforms.MelSpectrogram(
#     sample_rate=SAMPLE_RATE,
#     n_fft=4096,
#     win_length=3072,
#     hop_length=500,
#     n_mels=256
# )
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=1024,  # Reduce FFT size
    win_length=1024,  # Match window size to FFT size
    hop_length=64,  # Adjust hop size
    n_mels=256  # Reduce number of Mel bins
)

def process_audio(audio):
    """ Convert raw audio to model-compatible Mel Spectrogram """
    waveform = torch.tensor(audio).unsqueeze(0)  # Add batch dim
    if waveform.shape[-1] < 1024:  # Adjust to match `n_fft`
        print(f"Warning: Input waveform too short ({waveform.shape[-1]} samples). Padding...")
        waveform = torch.nn.functional.pad(waveform, (0, 1024 - waveform.shape[-1]))
    mel_spectrogram = mel_transform(waveform)
    log_mel = (mel_spectrogram + 1e-5).log()

    print(f"Original Mel Shape: {log_mel.shape}")  # Debugging output

    # 🔹 Ensure final shape is `[1, 1, 256, 65]`
    expected_shape = (1, 1, 256, 65)
    mel_spectrogram = F.interpolate(
        log_mel.unsqueeze(0), size=(256, 65), mode="bilinear", align_corners=False
    )

    print(f"Resized Mel Shape: {mel_spectrogram.shape}")  # Debugging output

    return log_mel.unsqueeze(0).numpy().astype(np.float32)  # Add batch dim & convert

# def process_audio(waveform):
#     print("Waveform shape:", waveform.shape)  # Debugging

#     # Ensure waveform has enough samples
#     if waveform.shape[0] < 4096:  # If it's smaller than `n_fft`
#         print("Warning: Input waveform too short for STFT!")
#         return None

#     mel_spectrogram = mel_transform(waveform)
#     return mel_spectrogram

def callback(indata, frames, time, status):
    """ Callback function for real-time streaming audio processing """
    if status:
        print(f"Error: {status}")
    if indata.shape[0] < SAMPLE_RATE * 2:  # Ensure at least 2 sec of audio
        print(f"Warning: Received {indata.shape[0]} samples, too short. Skipping frame.")
        return

    processed_audio = process_audio(indata[:, 0])  # Use first audio channel
    outputs = session.run(None, {input_name: processed_audio})
    print(f"Model Prediction: {outputs}")

# 🔹 Start real-time inference
with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback, blocksize=8192):
    print("Listening for streaming audio... Press Ctrl+C to stop.")
    while True:
        pass