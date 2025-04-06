import pyaudio
import wave
import argparse
import os
import time

def list_input_devices():
    """List all available input devices."""
    p = pyaudio.PyAudio()
    print("Available input devices:")
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if dev['maxInputChannels'] > 0:
            print(f"  [{i}] {dev['name']} - {dev['maxInputChannels']} channels - {dev['defaultSampleRate']} Hz")
    p.terminate()

def record_audio(filename: str, duration: int, channels: int = 4, rate: int = 44100, chunk: int = 4096, device_index: int = 1):
    """
    Records audio from the microphone and saves it to a WAV file.

    :param filename: The name of the output WAV file.
    :param duration: Duration of recording in seconds.
    :param channels: Number of audio channels (1 for mono, 2 for stereo).
    :param rate: Sampling rate in Hz.
    :param chunk: Buffer size.
    :param device_index: (Optional) Index of the input device.
    """
    # Initialize PyAudio
    p = pyaudio.PyAudio()

    if device_index is not None:
        try:
            device_info = p.get_device_info_by_index(device_index)
            print(f"Recording using device: {device_info['name']}")
        except IOError:
            print(f"Invalid device index: {device_index}. Using default device.")
            device_index = None
    else:
        print("Recording using default input device.")

    # Use default sampling rate if not specified
    if rate is None:
        rate = int(p.get_default_input_device_info()['defaultSampleRate'])
        print(f"Recording using a sampling rate of: {rate}")

    # Ensure the output directory exists
    recording_folder = os.path.dirname(os.path.abspath(filename))
    if not os.path.exists(recording_folder) and recording_folder != '':
        os.makedirs(recording_folder)

    print(f"Starting recording for {duration} seconds...")
    
    # Open the stream
    stream = p.open(format=pyaudio.paInt16,
                    channels=channels,
                    rate=rate,
                    input=True,
                    frames_per_buffer=chunk,
                    input_device_index=device_index)

    frames = []

    try:
        for _ in range(0, int(rate / chunk * duration)):
            data = stream.read(chunk, exception_on_overflow=False)
            frames.append(data)
    except KeyboardInterrupt:
        print("\nRecording interrupted by user.")
    except Exception as e:
        print(f"An error occurred during recording: {e}")
    finally:
        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        # Terminate PyAudio
        p.terminate()
        print("Recording finished.")

    # Save the recorded data as a WAV file
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))

    print(f"Audio saved to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record audio and save to a WAV file.")
    parser.add_argument('-l', '--list-devices', action='store_true',
                        help="List all available input devices and exit.")
    parser.add_argument('-f', '--file', type=str, default="output.wav",
                        help="Output WAV file name (default: output.wav)")
    parser.add_argument('-d', '--duration', type=int, default=300,
                        help="Recording duration in seconds (default: 5)")
    parser.add_argument('-c', '--channels', type=int, default=4,
                        help="Number of audio channels.")
    parser.add_argument('-r', '--rate', type=int, default=24000,
                        help="Sampling rate in Hz (default: 44100)")
    parser.add_argument('-k', '--chunk', type=int, default=500,
                        help="Buffer size (default: 4096)")
    parser.add_argument('-D', '--device', type=int, default=1,
                        help="Input device index (default: 1)")

    args = parser.parse_args()

    if args.list_devices:
        list_input_devices()
        exit(0)

    # print("Sleeping for a sec...")
    # time.sleep(15)
    # print("Done sleeping, starting recording!")

    for _ in range(10):
        fname = f"demo_amb_{_+1}.wav"
        print(f"Recording and saving file to {fname}")

        record_audio(
            filename=fname,
            duration=args.duration,
            channels=args.channels,
            rate=args.rate,
            chunk=args.chunk,
            device_index=args.device
        )