import pyaudio
import numpy as np
import wave
from scipy.io.wavfile import write
from scipy.io.wavfile import read
from scipy.signal import butter, lfilter

sample_rate = 48000 #Audio playback on Mac device works best at a 48 kHz sample rate.
high_amplitude = 32767
low_amplitude = -32768

preamble = "1010101010101010"# Known preamble for synchronization

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

    
    @staticmethod
    def getAmplitude(frames: list[int]) -> int:
        """ 
        calculate the root mean square (RMS) amplitude of audio frames,
        is useful for detecting when an audio signal starts and ends
        """
        # Convert frames to a numpy array
        frames = np.array(frames, dtype=np.float32)

        # Calculate RMS (Root Mean Square) value, which gives the amplitude
        rms = np.sqrt(np.mean(frames**2))
        
        # Convert to integer value if needed (keeping the scale intact)
        return int(rms)
         

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
    def __init__(self, baud_rate:int=300):
        self.__spaceTone = AFSKWaves.generateSpaceTone(baud_rate)
        self.__markTone = AFSKWaves.generateMarkTone(baud_rate)
        self.__byte_converter = ByteBitConverter()

    def prepare_for_sending(self, message:str):
        # Convert the message to binary bits
        data = message.encode('utf-8')
        bit_string = self.__byte_converter.bytesToBitStr(data)

        # Add a preamble (10 alternating bits) at beginning of each message
        bit_string = preamble + bit_string

        audio_data = []
        # Generate tones for each bit in the binary string
        for bit in bit_string:
            if bit == '0':
                audio_data.extend(self.__spaceTone)
            else:
                audio_data.extend(self.__markTone)
        
        # Convert the audio data to a numpy array and play using pyaudio
        audio_samples = np.array(audio_data, dtype=np.int16)
        return audio_samples
    
    """send text message via default loudspeaker """
    def send_msg(self, message: str):
        audio_samples = self.prepare_for_sending(message)
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
    
    """save the output audio into a .wav file"""
    def write_to_file(self, message:str, filename:str):
        audio_samples = self.prepare_for_sending(message)
        write(filename, sample_rate, audio_samples)



class Receiver:
    def __init__(self, baud_rate: int = 300, amp_start_threshold=10000, amp_end_threshold=8000 ):
        self.__space_tone: list[int] = AFSKWaves.generateSpaceTone(baud_rate)
        self.__mark_tone: list[int] = AFSKWaves.generateMarkTone(baud_rate)
        self.__byte_converter = ByteBitConverter()
        self.__amp_start_threshold: int = amp_start_threshold  # Example threshold for starting recording (adjust as needed)
        self.__amp_end_threshold:int = amp_end_threshold

    """remove noise outside the expected frequency range (below 1000 Hz or above 2500 Hz)"""
    def bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return lfilter(b, a, data)

    # it's correct, tolerance for frequency matching(150) can be adjusted
    def __reform_bit_String(self, audio, frames_per_bit: int) -> str:
        bit_string = ""
        fft_length = 2 ** int(np.ceil(np.log2(frames_per_bit)))

        for i in range(0, len(audio), frames_per_bit):
            chunk = audio[i:i+frames_per_bit]

            # Pad chunk with zeros if it's smaller than fft_length
            if len(chunk) < fft_length:
                chunk = np.pad(chunk, (0, fft_length - len(chunk)), mode='constant')
            
            # Perform FFT to identify dominant frequency
            fft_result = np.fft.fft(chunk, n=fft_length) #To increase frequency resolution, ensure the FFT input is long enough
            freqs = np.fft.fftfreq(len(chunk), 1 / sample_rate)
            fft_magnitude = np.abs(fft_result)
            
            # find the dominant frequency
            dominant_freq_index = np.argmax(fft_magnitude)
            dominant_freq = abs(freqs[dominant_freq_index])

            # Match dominant frequency to space (1200 Hz) or mark (2200 Hz)
            if abs(dominant_freq - 1200) < 150:#tolerance, if increase the tolerance from 100 to 150, there is no
                bit_string += "0"
            elif abs(dominant_freq - 2200) < 150:
                bit_string += "1"
        return bit_string
    
    def read_from_file(self, filename:str) -> str:
        sample_rate, audio_data = read(filename)
        
        # Verify mono or stereo
        if len(audio_data.shape) > 1:
            audio_data = audio_data[:, 0]  # Use only one channel for decoding
        
        baud_rate=300
        frames_per_bit = sample_rate // baud_rate # divide audio into chunks
        print("Frames per bit:", frames_per_bit)

        audio_data = self.bandpass_filter(audio_data, 1000, 2500, sample_rate)

        bit_string = self.__reform_bit_String(audio_data, frames_per_bit)

        # Check for the preamble in the bit string
        if preamble in bit_string:
            print("Preamble detected.")
            start_index = bit_string.index(preamble) + len(preamble)  # Start decoding after the preamble
            message_bits = bit_string[start_index:]
        else:
            print("Preamble not found in the audio. Still attempting to decode the entire bit string.")
            message_bits = bit_string

        # Convert the bit string to bytes and decode to a string
        try:
            original_message = self.__byte_converter.bitStrToBytes(message_bits).decode('utf-8', errors='ignore')
            return original_message
        except UnicodeDecodeError:
            print("Error decoding message from bit string.")
            return ""

        
    def receive_with_microphone_realtime(self, timeout: float, save_to_file:bool=False, output_filename: str="received_aduio.wav") -> str:
        """
        Listen to the microphone and decode the message in real-time
        """
        print("Listening for signal...")
        received_bits = ""
        audio_chunks = [] # for savinga audio to a file
        
        chunk_size = 1024 # Adjust this based on baud rate and frames per bit
        recorded_frames = 0
        max_frames = int(timeout * sample_rate)
        baud_rate = 300
        frames_per_bit = sample_rate // baud_rate

        # Initialize PyAudio stream
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk_size)

        try:
            while recorded_frames < max_frames:
                # Read a chunk of audio data
                audio_chunk = np.frombuffer(stream.read(chunk_size, exception_on_overflow=False), dtype=np.int16)
                recorded_frames += len(audio_chunk)

                if save_to_file:
                    audio_chunks.append(audio_chunk)

                # Apply bandpass filter to remove noise
                filtered_chunk = self.bandpass_filter(audio_chunk, 1000, 2500, sample_rate)
                
                # Decode bits from the chunk
                bit_string = self.__reform_bit_String(filtered_chunk, frames_per_bit)

                # Append decoded bits to the received_bits buffer
                received_bits += bit_string

                # Check if the preamble exists in the received bits
                #full_bit_string = "".join(received_bits)
                if preamble in received_bits:
                    print("Preamble detected. Decoding message...")
                    start_index = received_bits.index(preamble) + len(preamble)
                    message_bits = received_bits[start_index:]

                    # Try decoding the message as bytes
                    try:
                        decoded_message = self.__byte_converter.bitStrToBytes(message_bits).decode('utf-8', errors='ignore')
                        return decoded_message
                    except UnicodeDecodeError:
                        print("Decoding error. Retrying...")
                        continue

        finally:
            # Stop and close the stream
            stream.stop_stream()
            stream.close()
            p.terminate()

            # save the recorded audio to a file if enabled
            if save_to_file and audio_chunks:
                print(f"Saving recorded audio to {output_filename}...")
                with wave.open(output_filename, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
                    wf.setframerate(sample_rate)
                    wf.writeframes(b''.join(audio_chunks))

        print("Timeout: No signal detected.")
        return ""


# test_message = "Hello, my name is rui, how about you"*2
# sender = Sender()
# sender.send_msg(test_message)
# sender.write_to_file(test_message, "ouputAudio.wav")
receiver = Receiver()
#original_msg_file = receiver.read_from_file("ouputAudio.wav")
#print("original message from file: ",original_msg_file)

#original_msg = receiver.receive_with_microphone_realtime(5.0, save_to_file=True)
#print("message received with microphone: ", original_msg)
msg = receiver.read_from_file("received_aduio.wav")
print(msg)

