import pyaudio
import numpy as np
#from scipy.io.wavfile import write

sample_rate = 48000 #Audio playback on Mac device works best at a 48 kHz sample rate.
high_amplitude = 32767
low_amplitude = -32768

class AFSKWaves:

    @staticmethod
    def generateSpaceTone(baud_rate: int) -> list[int]: #baud rate:speed of modulation in bits per second
        """Generate the tone for a '0' (space) bit as a sine wave."""
        frames = sample_rate // baud_rate  # Number of samples per bit
        t = np.linspace(0, 1 / baud_rate, frames, endpoint=False)  # Time array for one bit
        frequency = 1200  # Frequency for '0' in Hz
        space_wave = high_amplitude * np.sin(2 * np.pi * frequency * t)  # Sine wave
        return space_wave.astype(np.int16)
    
    # calculates the number of frames needed for the mark frequency and generates the mark tone as a square wave directly
    @staticmethod
    def generateMarkTone(baud_rate: int) -> list[int]:
        """Generate the tone for a '1' (mark) bit as a sine wave."""
        frames = sample_rate // baud_rate  # Number of samples per bit
        t = np.linspace(0, 1 / baud_rate, frames, endpoint=False)  # Time array for one bit
        frequency = 2200  # Frequency for '1' in Hz
        mark_wave = high_amplitude * np.sin(2 * np.pi * frequency * t)  # Sine wave
        return mark_wave.astype(np.int16) 

    

class ByteBitConverter:
    # input_bytes= 'Hi', H (ASCII 72) is converted to '01001000', i (ASCII 105) is converted to '01101001'.
    # ''.join(...) then combines these binary strings into '0100100001101001'
    
    def bytesToBitStr(self, input_bytes: bytes) -> str:
        """converts each byte to an 8-bit binary string """
        return ''.join(f'{byte:08b}' for byte in input_bytes)

    def bitStrToBytes(self, input_bits: str) -> bytes:
        res = []
        for i in range(0, len(input_bits), 8):
            res.append(int(input_bits[i:i+8], 2))
        return bytes(res)


class Sender:
    def __init__(self, baut_rate:int=300):
        self.__spaceTone = AFSKWaves.generateSpaceTone(baut_rate)
        self.__markTone = AFSKWaves.generateMarkTone(baut_rate)
        self.__byte_converter = ByteBitConverter()
    
    def send_msg(self, message: str):
        # Convert the message to binary bits
        data = message.encode('utf-8')
        bit_string = self.__byte_converter.bytesToBitStr(data)
        audio_data = []
        # Generate tones for each bit in the binary string
        for bit in bit_string:
            if bit == '0':
                audio_data.extend(self.__spaceTone)
            else:
                audio_data.extend(self.__markTone)
        
        # Convert the audio data to a numpy array and play using pyaudio
        audio_samples = np.array(audio_data, dtype=np.int16)
        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=2,
            rate=sample_rate,
            output=True,
            output_device_index=1  # Use the default loudspeaker (index: 1)
        )
        # Play the audio
        stream.write(audio_samples.tobytes())

        # Cleanup
        stream.stop_stream()
        stream.close()
        pa.terminate()
    


test_message = "Hello, my name is rui, how about you"*2
sender = Sender()
sender.send_msg(test_message)
