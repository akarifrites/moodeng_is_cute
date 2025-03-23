import pyaudio

def list_microphones():
    p = pyaudio.PyAudio()
    print("Available Microphones:")
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        if device_info["maxInputChannels"] > 0:  # Ensure it's an input device
            print(f"Index {i}: {device_info['name']}")
    p.terminate()

if __name__ == "__main__":
    list_microphones()