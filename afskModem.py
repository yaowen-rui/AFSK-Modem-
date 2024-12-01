import pyaudio
import numpy as np
from scipy.io.wavfile import write
from scipy.io.wavfile import read
from scipy.signal import butter, lfilter

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
    def __init__(self, baud_rate: int = 300, amp_start_threshold=14000, amp_end_threshold=9000 ):
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
        for i in range(0, len(audio), frames_per_bit):
            chunk = audio[i:i+frames_per_bit]

            # Perform FFT to identify dominant frequency
            fft_result = np.fft.fft(chunk)
            freqs = np.fft.fftfreq(len(chunk), 1 / sample_rate)
            dominant_freq = abs(freqs[np.argmax(np.abs(fft_result))])

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

        # Convert bit string to bytes
        original_message = self.__byte_converter.bitStrToBytes(bit_string).decode('utf-8')
        return original_message

    """ record the audio for a specified timeout duration(second) from the microphone """
    # something wrong， can't detect the siganl with mircophone
    def __listen(self, timeout: float) ->list[int]:
        recorded_frames: list[int] = []
        listened_frames: int = 0
        timeout_frames = int(timeout * sample_rate )  # Convert timeout to frame count
        
        #initialize pyaudio and microphone input stream
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=1024)
        print("Listening for start signal...")
        
        #listen for the start of the signal
        while listened_frames < timeout_frames:
            frames = np.frombuffer(stream.read(1024, exception_on_overflow=False), dtype=np.int16)
            if AFSKWaves.getAmplitude(frames) > self.__amp_start_threshold:
                print("Signal detected, recording started")
                recorded_frames.extend(frames)  # Start recording
                break
            listened_frames += 1024

        # Continue recording until the signal falls below the end threshold
        while True:
            frames = np.frombuffer(stream.read(1024, exception_on_overflow=False), dtype=np.int16)
            recorded_frames.extend(frames)
            if AFSKWaves.getAmplitude(frames) < self.__amp_end_threshold:
                print("Signal ended, recording stopped")
                break
        
        # If timeout reached without detecting a start signal, return empty
        if listened_frames >= timeout_frames:
            print("Timeout: No signal detected")
            stream.stop_stream()
            stream.close()
            return []

        # Stop the stream and close it
        stream.stop_stream()
        stream.close()
        return recorded_frames
        
    def receive_with_microphone(self, timeout:float) -> str:
        received_audio = self.__listen(timeout)
        if received_audio == []:
            print("Time out, no data received!")
            return ""
        
        # Convert audio data to a numpy array and apply the same processing as in file-based reception
        baud_rate = 300
        frames_per_bit = sample_rate // baud_rate

        received_audio = self.bandpass_filter(received_audio, 1000, 2500, sample_rate)

        bit_string = self.__reform_bit_String(received_audio, frames_per_bit)       

        # Decode the bit string into bytes and then to a string
        received_message = self.__byte_converter.bitStrToBytes(bit_string).decode('utf-8')
        
        return received_message

# test_message = "Hello, my name is rui, how about you"*2
# sender = Sender()
# sender.send_msg(test_message)
# sender.write_to_file(test_message, "ouputAudio.wav")
receiver = Receiver()
#original_msg_file = receiver.read_from_file("ouputAudio.wav")
#print("original message from file: ",original_msg_file)

original_msg = receiver.receive_with_microphone(timeout=10.0)
print("message received with microphone: ", original_msg)

