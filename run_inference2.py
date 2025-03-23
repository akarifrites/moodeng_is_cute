import numpy as np
import pyaudio
import onnxruntime
import time
import threading
import queue
import librosa

CHUNK = 1024               # Number of audio samples per frame
FORMAT = pyaudio.paFloat32 # Audio format
CHANNELS = 1               # Mono audio
RATE = 32000               # Sample rate in Hz (adjust based on your model)
BUFFER_DURATION = 1        # Seconds of audio to accumulate before inference
SAMPLES_REQUIRED = RATE * BUFFER_DURATION

audio_queue = queue.Queue() # Thread-safe Queue to hold audio data

running = True

# Audio Callback Function 
def audio_callback(in_data, frame_count, time_info, status):
    # Convert raw bytes to a numpy array (float32)
    audio_data = np.frombuffer(in_data, dtype=np.float32)
    audio_queue.put(audio_data)
    return (None, pyaudio.paContinue)

def get_mel_spectrogram(audio_data, sample_rate=32000):
    # Compute the mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio_data, sr=sample_rate, n_mels=256, fmax=8000, n_fft=2048, hop_length=512
    )
    # Convert to log scale (dB)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
    # Normalize between 0 and 1
    norm_mel_spectrogram = (log_mel_spectrogram - log_mel_spectrogram.min()) / (
        log_mel_spectrogram.max() - log_mel_spectrogram.min()
    )
    # Ensure the output has exactly 65 time frames
    norm_mel_spectrogram = librosa.util.fix_length(norm_mel_spectrogram, size=65, axis=1)
    return norm_mel_spectrogram

def inference_loop(session, labels, stream):
    
    audio_buffer = np.array([], dtype=np.float32)

    while running:
        try:
        #     # Read a chunk of audio data
        #     data = stream.read(CHUNK, exception_on_overflow=False)
        # except Exception as e:
        #     print("Error reading audio:", e)
        #     continue
            # Wait for new audio data to arrive
            data = audio_queue.get(timeout=1)
            audio_buffer = np.concatenate((audio_buffer, data))
        except queue.Empty:
            continue

        # Convert raw bytes to a NumPy array
        # audio_data = np.frombuffer(data, dtype=np.float32)
        
        if len(audio_buffer) >= SAMPLES_REQUIRED:
            input_data = audio_buffer[:SAMPLES_REQUIRED] # Extract a segment of the required length
            audio_buffer = audio_buffer[SAMPLES_REQUIRED:] # Remove the used samples from the buffer (for overlapping, change this logic)

            mel_spec = get_mel_spectrogram(input_data, sample_rate=RATE)
            input_tensor = mel_spec[np.newaxis, np.newaxis, :, :] # Shape: [1, 1, 256, 65]
            
            input_name = session.get_inputs()[0].name
            # print("Model input shape:", session.get_inputs()[0].shape)
            result = session.run(None, {input_name: input_tensor})

            # Assuming the model outputs logits or probabilities for each class
            probabilities = np.squeeze(result[0])  # Shape: [10]

            print("Classification results:")
            for i, label in enumerate(labels):
                score = probabilities[i]
                print(f"{label}: {score:.2f}", end='\t')
            print("")

            # threshold = 0.7
            # for i, label in enumerate(labels):
            #     if probabilities[i] >= threshold:
            #         print(f"Action for {label} detected with probability {probabilities[i]:.2f}")
            
            # Process or print the inference result
            print("Inference result:", result)
            
        time.sleep(0.01)

def main():
    global running
    labels = [
        "airport", "shopping_mall", "metro_station", "street_pedestrian", 
        "public_square", "street_traffic", "tram", "bus", "metro", "park"
    ]

    # Load the ONNX model (make sure "model.onnx" is in your working directory or provide the correct path)
    session = onnxruntime.InferenceSession("baseline.onnx")
    print("Model Loaded Successfully")
    
    # Initialize PyAudio and open the stream
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    # input_device_index=27,
                    frames_per_buffer=CHUNK,
                    stream_callback=audio_callback)
    
    stream.start_stream()
    print("Streaming audio... Press Ctrl+C to stop.")

    # Start the inference loop in a separate thread
    inference_thread = threading.Thread(target=inference_loop, args=(session, labels, stream))
    inference_thread.daemon = True
    inference_thread.start()
    
    try:
        while stream.is_active():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Stopping...")
    
    # Cleanup
    stream.stop_stream()
    stream.close()
    p.terminate()

if __name__ == "__main__":
    main()