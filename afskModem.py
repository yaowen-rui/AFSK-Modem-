import numpy as np
from scipy.io.wavfile import write, read
from scipy.signal import butter, lfilter
import math

# Global configuration
sample_rate = 48000  # Sampling frequency
high_amplitude = 32767
preamble = "1010101010101010"  # Preamble for synchronization

# Class to generate AFSK tones
class AFSKWaves:

    @staticmethod
    def generate_space_tone(baud_rate: int) -> np.ndarray:
        frames = sample_rate // baud_rate
        t = np.linspace(0, 1 / baud_rate, frames, endpoint=False)
        frequency = 1000
        space_wave = high_amplitude * np.sin(2 * np.pi * frequency * t)
        return space_wave.astype(np.int16)

    @staticmethod
    def generate_mark_tone(baud_rate: int) -> np.ndarray:
        frames = sample_rate // baud_rate
        t = np.linspace(0, 1 / baud_rate, frames, endpoint=False)
        frequency = 2200
        mark_wave = high_amplitude * np.sin(2 * np.pi * frequency * t)
        return mark_wave.astype(np.int16)

# Class to convert between bytes and bit strings
class ByteBitConverter:

    def bytes_to_bit_str(self, input_bytes: bytes) -> str:
        return ''.join(f'{byte:08b}' for byte in input_bytes)

    def bit_str_to_bytes(self, input_bits: str) -> bytes:
        res = []
        for i in range(0, len(input_bits), 8):
            byte_bits = input_bits[i:i+8]
            if len(byte_bits) == 8:
                res.append(int(byte_bits, 2))
        return bytes(res)

# Sender class
class Sender:
    def __init__(self, baud_rate: int = 300):
        self.space_tone = AFSKWaves.generate_space_tone(baud_rate)
        self.mark_tone = AFSKWaves.generate_mark_tone(baud_rate)
        self.byte_converter = ByteBitConverter()
        self.baud_rate = baud_rate

    def prepare_for_sending(self, message: str):
        data = message.encode('utf-8')
        bit_string = self.byte_converter.bytes_to_bit_str(data)
        bit_string = preamble + bit_string

        audio_data = []
        for bit in bit_string:
            if bit == '0':
                audio_data.extend(self.space_tone)
            else:
                audio_data.extend(self.mark_tone)

        audio_samples = np.array(audio_data, dtype=np.int16)
        return audio_samples

    def write_to_file(self, message: str, filename: str):
        audio_samples = self.prepare_for_sending(message)
        write(filename, sample_rate, audio_samples)
        print(f"Sender: Message saved to '{filename}'.")

# Receiver class
class Receiver:
    def __init__(self, baud_rate: int = 300):
        self.space_tone = AFSKWaves.generate_space_tone(baud_rate)
        self.mark_tone = AFSKWaves.generate_mark_tone(baud_rate)
        self.byte_converter = ByteBitConverter()
        self.baud_rate = baud_rate
        self.received_message = ''

    def bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return lfilter(b, a, data)

    def __reform_bit_string(self, audio, frames_per_bit: int) -> str:
        bit_string = ""
        for i in range(0, len(audio), frames_per_bit):
            chunk = audio[i:i+frames_per_bit]
            if len(chunk) == 0:
                continue
            fft_result = np.fft.fft(chunk)
            freqs = np.fft.fftfreq(len(chunk), 1 / sample_rate)
            dominant_freq = abs(freqs[np.argmax(np.abs(fft_result))])
            if abs(dominant_freq - 1000) < 400:
                bit_string += "0"
            elif abs(dominant_freq - 2200) < 400:
                bit_string += "1"
        return bit_string

    def read_from_file(self, filename: str) -> str:
        print("Receiver: Reading from file...")
        sample_rate_file, audio_data = read(filename)
        if sample_rate_file != sample_rate:
            print("Warning: Sample rate does not match.")

        if len(audio_data.shape) > 1:
            audio_data = audio_data[:, 0]

        frames_per_bit = sample_rate // self.baud_rate
        print("Frames per bit:", frames_per_bit)

        audio_data = self.bandpass_filter(audio_data, 900, 2500, sample_rate)
        bit_string = self.__reform_bit_string(audio_data, frames_per_bit)

        # Debug: Print the received bit string
        print(f"Received bit string: {bit_string}")

        if preamble in bit_string:
            print("Receiver: Preamble detected. Decoding message...")
            start_index = bit_string.index(preamble) + len(preamble)
            message_bits = bit_string[start_index:]

            try:
                original_message = self.byte_converter.bit_str_to_bytes(message_bits).decode('utf-8', errors='ignore')
                self.received_message = original_message
                return original_message
            except UnicodeDecodeError:
                print("Error decoding the message.")
                self.received_message = ''
                return ''
        else:
            print("Receiver: Preamble not found.")
            self.received_message = ''
            return ''

if __name__ == "__main__":
    # Create and use the sender
    test_message = "am Testen, ob das funktioniert."
    sender = Sender()

    # Save the message to a WAV file
    sender.write_to_file(test_message, "output_audio.wav")

    # Create the receiver
    receiver = Receiver()

    # Read and decode the message from the WAV file
    received_message = receiver.read_from_file("output_audio.wav")
    print("Message received from file:", received_message)
