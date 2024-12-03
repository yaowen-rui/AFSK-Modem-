import pyaudio
import numpy as np
from scipy.io.wavfile import write
import time

# Global configuration
SAMPLE_RATE = 48000  # Sampling frequency
AMPLITUDE = 32767
PREAMBLE = "1010101010101010"  # Preamble for synchronization
BAUD_RATE = 300  # Bits per second

class AFSKWaves:
    @staticmethod
    def generate_tone(frequency: int, duration: float) -> np.ndarray:
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
        wave = AMPLITUDE * np.sin(2 * np.pi * frequency * t)
        return wave.astype(np.int16)

    @staticmethod
    def generate_bit_wave(bit: str) -> np.ndarray:
        if bit == '0':
            frequency = 1200  # Frequency for '0'
        else:
            frequency = 2200  # Frequency for '1'
        duration = 1 / BAUD_RATE
        return AFSKWaves.generate_tone(frequency, duration)

class ByteBitConverter:
    def bytes_to_bit_str(self, input_bytes: bytes) -> str:
        return ''.join(f'{byte:08b}' for byte in input_bytes)

class Sender:
    def __init__(self):
        self.byte_converter = ByteBitConverter()

    def prepare_signal(self, message: str) -> np.ndarray:
        data = message.encode('utf-8')
        bit_string = PREAMBLE + self.byte_converter.bytes_to_bit_str(data)

        audio_data = []
        for bit in bit_string:
            bit_wave = AFSKWaves.generate_bit_wave(bit)
            audio_data.append(bit_wave)

        audio_signal = np.concatenate(audio_data)
        return audio_signal

    def send(self, message: str):
        audio_signal = self.prepare_signal(message)
        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            output=True
        )
        print("Transmitter: Sending message...")
        stream.write(audio_signal.tobytes())
        stream.stop_stream()
        stream.close()
        pa.terminate()
        print("Transmitter: Message sent.")

if __name__ == "__main__":
    test_message = "Hello, this is a test message."
    sender = Sender()
    time.sleep(2)  # Wait for 2 seconds before sending
    sender.send(test_message)
