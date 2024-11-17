import numpy as np
from scipy.io.wavfile import write
import matplotlib.pyplot as plt


sample_rate = 48000  
high_amplitude = 32767  
low_amplitude = -32768  


class AFSKWaves:
    @staticmethod
    def generateSpaceTone(baud_rate: int) -> np.ndarray:
        """Generate the tone for a '0' (space) bit as a sine wave."""
        frames = sample_rate // baud_rate  # Number of samples per bit
        t = np.linspace(0, 1 / baud_rate, frames, endpoint=False)  # Time array for one bit
        frequency = 1200  # Frequency for '0' in Hz
        space_wave = high_amplitude * np.sin(2 * np.pi * frequency * t)  # Sine wave
        return space_wave.astype(np.int16)

    @staticmethod
    def generateMarkTone(baud_rate: int) -> np.ndarray:
        """Generate the tone for a '1' (mark) bit as a sine wave."""
        frames = sample_rate // baud_rate  # Number of samples per bit
        t = np.linspace(0, 1 / baud_rate, frames, endpoint=False)  # Time array for one bit
        frequency = 2200  # Frequency for '1' in Hz
        mark_wave = high_amplitude * np.sin(2 * np.pi * frequency * t)  # Sine wave
        return mark_wave.astype(np.int16)


class ByteBitConverter:
    def bytesToBitStr(self, input_bytes: bytes) -> str:
        """Convert bytes into a binary string."""
        return ''.join(f'{byte:08b}' for byte in input_bytes)


def test_afsk_with_msg(message: str, baud_rate: int = 300, output_file: str = "afsk_output.wav"):
    """
    Convert a text message into an AFSK-modulated audio signal and save it as a .wav file.
    
    Args:
        message (str): The text message to be converted into audio.
        baud_rate (int): The baud rate (bits per second) for modulation.
        output_file (str): The name of the output .wav file.
    """
    print(f"Generating AFSK signal for the message: '{message}' with baud rate: {baud_rate}")
    
    # Convert the message into a binary string
    byte_converter = ByteBitConverter()
    data = message.encode('utf-8')  # Encode the message into bytes
    bit_string = byte_converter.bytesToBitStr(data)  # Convert bytes to binary string
    
    # Generate audio tones for each bit
    audio_data = []
    for bit in bit_string:
        if bit == '0':
            audio_data.extend(AFSKWaves.generateSpaceTone(baud_rate))
        else:
            audio_data.extend(AFSKWaves.generateMarkTone(baud_rate))

    # Convert the generated signal into a NumPy array
    audio_samples = np.array(audio_data, dtype=np.int16)
    
    # Calculate the duration of the generated audio
    duration = len(audio_samples) / sample_rate
    print(f"Generated audio duration: {duration:.2f} seconds")

    # Save the audio signal as a .wav file
    write(output_file, sample_rate, audio_samples)
    print(f"Audio file saved as '{output_file}'")

    # Visualize the generated signal
    plt.plot(audio_samples[:1000])  # Plot the first 1000 samples
    plt.title("Generated AFSK Audio Signal")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.show()


# Test with a longer, repeated message
test_message = "Hello, this is a test message. " * 5  
test_afsk_with_msg(test_message, baud_rate=300)  # Use a lower baud rate
