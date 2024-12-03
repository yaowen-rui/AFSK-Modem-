import numpy as np
import pyaudio
import threading
import time
from scipy.signal import butter, lfilter

# Configuration parameters
SAMPLE_RATE = 48000  # Sampling rate
BAUD_RATE = 300      # Baud rate
HIGH_AMPLITUDE = 32767
LOW_AMPLITUDE = -32768
PREAMBLE = '1010101010101010'  # Preamble for synchronization
FREQ_ZERO = 1200    # Frequency for bit '0'
FREQ_ONE = 2200     # Frequency for bit '1'
TOLERANCE = 200     # Frequency detection tolerance
AMP_START_THRESHOLD = 10000  # Signal start threshold
AMP_END_THRESHOLD = 5000     # Signal end threshold

# Functions to generate AFSK tones
def generate_tone(frequency, duration):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    wave = HIGH_AMPLITUDE * np.sin(2 * np.pi * frequency * t)
    return wave.astype(np.int16)

def generate_afsk_signal(bit_string):
    bit_duration = 1 / BAUD_RATE
    signal = np.array([], dtype=np.int16)
    for bit in bit_string:
        if bit == '0':
            tone = generate_tone(FREQ_ZERO, bit_duration)
        else:
            tone = generate_tone(FREQ_ONE, bit_duration)
        signal = np.concatenate((signal, tone))
    return signal

# Utility functions
def bytes_to_bitstring(data_bytes):
    return ''.join(f'{byte:08b}' for byte in data_bytes)

def bitstring_to_bytes(bit_string):
    bytes_list = [int(bit_string[i:i+8], 2) for i in range(0, len(bit_string), 8)]
    return bytes(bytes_list)

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

# Transmitter class
class Sender:
    def __init__(self, message):
        self.message = message
        self.bit_string = PREAMBLE + bytes_to_bitstring(message.encode('utf-8'))
        self.signal = generate_afsk_signal(self.bit_string)

    def send(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=SAMPLE_RATE,
                        output=True)
        stream.write(self.signal.tobytes())
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("Transmitter: Message sent.")

# Receiver class
class Receiver:
    def __init__(self, input_device_index=None):
        self.received_bits = ''
        self.input_device_index = input_device_index

    def listen(self, duration):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=SAMPLE_RATE,
                        input=True,
                        frames_per_buffer=1024,
                        input_device_index=self.input_device_index)
        frames = []
        start_time = time.time()
        recording = False
import pyaudio
import numpy as np
from scipy.signal import butter, lfilter
import threading
import time
import math

# Global configuration
SAMPLE_RATE = 48000  # Sampling frequency
PREAMBLE = "1010101010101010"  # Preamble for synchronization
BAUD_RATE = 300  # Bits per second
AMP_START_THRESHOLD = 500  # Adjust as needed
AMP_END_THRESHOLD = 200  # Adjust as needed

class ByteBitConverter:
    def bit_str_to_bytes(self, input_bits: str) -> bytes:
        res = []
        for i in range(0, len(input_bits), 8):
            byte_bits = input_bits[i:i+8]
            if len(byte_bits) == 8:
                res.append(int(byte_bits, 2))
        return bytes(res)

class Receiver:
    def __init__(self, input_device_index=None):
        self.byte_converter = ByteBitConverter()
        self.input_device_index = input_device_index
        self.received_message = ''

    def bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return lfilter(b, a, data)

    def detect_bits(self, audio_data, frames_per_bit):
        bit_string = ""
        for i in range(0, len(audio_data), frames_per_bit):
            chunk = audio_data[i:i+frames_per_bit]
            if len(chunk) == 0:
                continue
            fft_result = np.fft.fft(chunk)
            freqs = np.fft.fftfreq(len(chunk), 1 / SAMPLE_RATE)
            dominant_freq = abs(freqs[np.argmax(np.abs(fft_result))])
            if abs(dominant_freq - 1200) < 400:
                bit_string += "0"
            elif abs(dominant_freq - 2200) < 400:
                bit_string += "1"
            else:
                bit_string += ""  # Ignore unrecognized frequencies
        return bit_string

    def receive(self, timeout: float):
        print("Receiver: Listening...")
        chunk_size = 1024
        frames_per_bit = int(SAMPLE_RATE / BAUD_RATE)
        recording = False
        recorded_data = []

        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=SAMPLE_RATE,
                        input=True,
                        frames_per_buffer=chunk_size,
                        input_device_index=self.input_device_index)
        try:
            start_time = time.time()
            while time.time() - start_time < timeout:
                data = stream.read(chunk_size, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.int16)
                if len(audio_chunk) == 0:
                    continue
                amplitude = np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2))
                if not recording and amplitude > AMP_START_THRESHOLD:
                    recording = True
                    print("Receiver: Signal detected, recording...")
                if recording:
                    recorded_data.extend(audio_chunk)
                    if amplitude < AMP_END_THRESHOLD:
                        print("Receiver: Signal ended.")
                        break
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

        if recorded_data:
            audio_data = np.array(recorded_data, dtype=np.int16)
            filtered_data = self.bandpass_filter(audio_data, 1000, 2500, SAMPLE_RATE)
            bit_string = self.detect_bits(filtered_data, frames_per_bit)
            print(f"Receiver: Received bit string: {bit_string}")
            if PREAMBLE in bit_string:
                print("Receiver: Preamble detected. Decoding message...")
                start_index = bit_string.index(PREAMBLE) + len(PREAMBLE)
                message_bits = bit_string[start_index:]
                try:
                    original_message = self.byte_converter.bit_str_to_bytes(message_bits).decode('utf-8', errors='ignore')
                    self.received_message = original_message
                except UnicodeDecodeError:
                    print("Receiver: Error decoding message.")
                    self.received_message = ''
            else:
                print("Receiver: Preamble not found.")
                self.received_message = ''
        else:
            print("Receiver: No data recorded.")

if __name__ == "__main__":
    # List available input devices
    p = pyaudio.PyAudio()
    print("Available audio input devices:")
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        if device_info.get('maxInputChannels') > 0:
            print(f"Device ID {i}: {device_info.get('name')}")
    p.terminate()

    # Prompt user for device ID
    input_device_id = int(input("Enter the Device ID of your microphone: "))

    receiver = Receiver(input_device_index=input_device_id)

    # Start receiver in a separate thread
    receiver_thread = threading.Thread(target=receiver.receive, args=(15.0,))
    receiver_thread.start()

    # Wait for the receiver to finish
    receiver_thread.join()

    print("Message received:", receiver.received_message)
