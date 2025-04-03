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
RATE = 44100               # Sample rate in Hz (adjust based on your model)
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

def get_mel_spectrogram(audio_data, sample_rate=RATE):
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

def is_audio_signal_present(audio_data, threshold=0.01):
    """
    Compute the RMS energy of the audio and compare it to a threshold.
    Adjust the threshold based on your environment.
    """
    rms = np.sqrt(np.mean(audio_data**2))
    return rms > threshold

def inference_loop(session, labels, stream):
    audio_buffer = np.array([], dtype=np.float32)
    listening = False  # Flag to indicate if significant audio is present

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

            # Set the listening flag based on the RMS energy of the input
            listening = is_audio_signal_present(input_data, threshold=0.01)
            if not listening:
                print("Background noise detected. Skipping classification.")
                continue  # Skip classification if the audio signal is too weak

            mel_spec = get_mel_spectrogram(input_data, sample_rate=RATE)
            input_tensor = mel_spec[np.newaxis, np.newaxis, :, :] # Shape: [1, 1, 256, 65]
            
            input_name = session.get_inputs()[0].name
            # print("Model input shape:", session.get_inputs()[0].shape)
            result = session.run(None, {input_name: input_tensor})

            # Assuming the model outputs logits or probabilities for each class
            logits = np.squeeze(result[0])  # Shape: [10]
            probabilities = np.exp(logits) / np.sum(np.exp(logits))

            print("Classification results:")
            max_label = None
            max_score = -float('inf')
            for i, label in enumerate(labels):
                    score = probabilities[i]
                    print(f"{label}: {score:.2f}", end='\t')
                    if score > max_score:  # Check if this score is the highest so far
                        max_score = score
                        max_label = label
            print("")
            print(f"classified as: {max_label} with score {max_score:.2f}")

            # threshold = 0.7
            # for i, label in enumerate(labels):
            #     if probabilities[i] >= threshold:
            #         print(f"Action for {label} detected with probability {probabilities[i]:.2f}")
            
            # Process or print the inference result
            print("Inference result:", result)
            
        time.sleep(0.01)

def main():
    global running
    labels = ["airport", "shopping_mall", "metro_station", "street_pedestrian", 
              "public_square", "street_traffic", "tram", "bus", "metro", "park"
    ]

    model_path = r"C:\Users\fenel\Documents\dcase2024_task1_baseline\onnx_runtime\baseline.onnx"
    session = onnxruntime.InferenceSession(model_path)
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