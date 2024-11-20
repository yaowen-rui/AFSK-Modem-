import pyaudio
import numpy as np
from scipy.io.wavfile import write

sample_rate = 48000
high_amplitude = 32767
low_amplitude = -32768

class AFSKWaves:

    # space:binary 0, generate a tone for space frequency
    @staticmethod
    def generateSpaceTone(baud_rate: int) -> list[int]: #baud rate:speed of modulation in bits per second

        if(sample_rate % baud_rate != 0): #It can use any baud rate as long as it is a factor of the sample rate
            raise Exception("Baud rate should be a factor of sample rate 48000.")

        frames = sample_rate // baud_rate # frames: the number of audio samples per bit

        # generate square wave for half of frames with a high amplitude, and for the other half with a low amplitude
        # this forms a square wave at space frequency
        square_wave = np.tile([high_amplitude, low_amplitude], frames // 2) #generates an alternating pattern [32767, -32768], repeated frames // 2 times.
    
        return square_wave.astype(int).tolist() #Output: A list of integers representing the samples of a square wave at the "space" frequency.
    
    # mark:binary 1, generate a tone for mark frequency
    # calculates the number of frames needed for the mark frequency and generates the mark tone as a square wave directly
    @staticmethod
    def generateMarkTone(baud_rate: int) -> list[int]:
        if(sample_rate % baud_rate != 0): 
            raise Exception("Baud rate should be a factor of sample rate 48000.")
        
        # Calculate frames per mark tone cycle (using twice the baud rate frequency)
        frames = sample_rate // (baud_rate * 2)
        res = []
        # Generate one cycle of the mark tone
        for i in range(2 * frames):
            if i < frames:
                if i < frames // 2:
                    res.append(high_amplitude)  # First half of the wave (positive amplitude)
                else:
                    res.append(low_amplitude)  # Second half of the wave (negative amplitude)
            else:
                if i < frames + (frames // 2):
                    res.append(high_amplitude)  # Second cycle, first half (positive amplitude)
                else:
                    res.append(low_amplitude)  # Second cycle, second half (negative amplitude)

        return res 

    

class ByteBitConverter:
    # converts each byte to an 8-bit binary string
    # input_bytes= 'Hi', H (ASCII 72) is converted to '01001000', i (ASCII 105) is converted to '01101001'.
    # ''.join(...) then combines these binary strings into '0100100001101001'
    
    def bytesToBitStr(self, input_bytes: bytes) -> str:
        return ''.join(f'{byte:08b}' for byte in input_bytes)

    
    def bitStrToBytes(self, input_bits: str) -> bytes:
        res = []
        for i in range(0, len(input_bits), 8):
            res.append(int(input_bits[i:i+8], 2))
        return bytes(res)

def test_afsk_with_msg(message:str, baud_rate: int= 300, output_file: str="testAudioOutput.wav"):
    # Convert the message to binary bits
    byte_converter = ByteBitConverter()
    
    if isinstance(message, str):
            data = message.encode('utf-8')
    bit_string = byte_converter.bytesToBitStr(data)

    # Generate tones for each bit in the binary string
    
    audio_data = []

    for bit in bit_string:
        if bit == '0':
            audio_data.extend(AFSKWaves.generateSpaceTone(baud_rate))
        else:
            audio_data.extend(AFSKWaves.generateMarkTone(baud_rate))

    # Convert the audio data to a numpy array and play using pyaudio
    audio_samples = np.array(audio_data, dtype=np.int16)
    
    write(output_file, sample_rate, audio_samples)

    # # Set up pyaudio to play the generated tone
    # p = pyaudio.PyAudio()
    # # Print available devices
    # for i in range(p.get_device_count()):
    #     device_info = p.get_device_info_by_index(i)
    #     print(f"Device {i}: {device_info['name']}")

    # # Select the desired device index
    # device_index = 1  # Replace with the index of your preferred output device
    
    # # Open the stream with an explicit device index
    # stream = p.open(format=pyaudio.paInt16,
    #                 channels=1,
    #                 rate=sample_rate,
    #                 output=True,
    #                 output_device_index=device_index)
    # print(f"Playing AFSK modulated message: {message}")
    # stream.write(audio_samples.tobytes())

    # # Clean up the stream and pyaudio
    # stream.stop_stream()
    # stream.close()
    # p.terminate()

# Test with a sample message, we can hear the AFSK modulated tones for the test msg
# To visualize the signal further, you can save audio_samples as a .wav file 
# and open it in a tool like GNURadio for a spectrogram analysis.
test_message = "Hello, my name is rui, how about you"
test_afsk_with_msg(test_message, baud_rate=300)