import pyaudio
import numpy as np
import wave
import matplotlib.pyplot as plt
from scipy.io.wavfile import write, read
from scipy.signal import butter, lfilter, correlate



sample_rate = 48000
high_amplitude = 32767
low_amplitude = -32768

sync_frequency = 3000
sync_duration = 3.0

class AFSKWaves:

    @staticmethod
    def generateSpaceTone(baud_rate: int) -> np.ndarray:
        frames = sample_rate // baud_rate
        t = np.linspace(0, 1 / baud_rate, frames, endpoint=False)
        frequency = 1200
        space_wave = high_amplitude * np.sin(2 * np.pi * frequency * t)
        return space_wave.astype(np.int16)
    
    @staticmethod
    def generateMarkTone(baud_rate: int) -> np.ndarray:
        frames = sample_rate // baud_rate
        t = np.linspace(0, 1 / baud_rate, frames, endpoint=False)
        frequency = 2200
        mark_wave = high_amplitude * np.sin(2 * np.pi * frequency * t)
        return mark_wave.astype(np.int16) 
    
    @staticmethod
    def generateSyncTone(baut_rate:int) -> np.ndarray:
        t = np.linspace(0, sync_duration, int(sample_rate * sync_duration), endpoint=False)
        sync_tone = high_amplitude * np.sin(2 * np.pi * sync_frequency * t)
        return sync_tone.astype(np.int16)


class ByteBitConverter:
    def bytesToBitStr(self, input_bytes: bytes) -> str:
        return ''.join(f'{byte:08b}' for byte in input_bytes)

    def bitStrToBytes(self, input_bits: str) -> bytes:
        res = []
        for i in range(0, len(input_bits), 8):
            byte_str = input_bits[i:i+8]
            if len(byte_str) < 8:
                # Pad with zeros if needed
                byte_str = byte_str.ljust(8, '0')
            res.append(int(byte_str, 2))
        return bytes(res)

class Sender:
    def __init__(self, baud_rate: int = 300):
        self.baud_rate = baud_rate
        self.__spaceTone = AFSKWaves.generateSpaceTone(baud_rate)
        self.__markTone = AFSKWaves.generateMarkTone(baud_rate)
        self.__syncTone = AFSKWaves.generateSyncTone(baud_rate)
        self.__byte_converter = ByteBitConverter()

    def prepare_for_sending(self, message: str):
        data = message.encode('utf-8')
        bit_string = self.__byte_converter.bytesToBitStr(data)

        audio_data = []
        for bit in bit_string:
            if bit == '0':
                audio_data.extend(self.__spaceTone)
            else:
                audio_data.extend(self.__markTone)

        message_audio = np.array(audio_data, dtype=np.int16)
        full_audio_signal = np.concatenate([self.__syncTone, message_audio])
        return full_audio_signal
    
    def send_msg(self, message: str):
        audio_samples = self.prepare_for_sending(message)
        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=2,
            rate=sample_rate,
            output=True,
            output_device_index=1
        )
        stream.write(audio_samples.tobytes())
        stream.stop_stream()
        stream.close()
        pa.terminate()
    
    def write_to_file(self, message:str, filename:str):
        audio_samples = self.prepare_for_sending(message)
        write(filename, sample_rate, audio_samples)



class Receiver:
    def __init__(self, baud_rate: int = 300):
        self.baud_rate = baud_rate
        self.__space_tone = AFSKWaves.generateSpaceTone(baud_rate)
        self.__mark_tone = AFSKWaves.generateMarkTone(baud_rate)
        self.__sync_tone = AFSKWaves.generateSyncTone(baud_rate)

        self.__byte_converter = ByteBitConverter()
        self.frames_per_bit = sample_rate // baud_rate

    def bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return lfilter(b, a, data)

    def detect_sync_tone(self, audio_chunk, tolerance=150)->bool:
        """Detect the synchronization tone."""
        if len(audio_chunk) == 0:
            return False
        # Normalize the audio chunk to avoid amplitude issues
        audio_chunk = audio_chunk / (np.max(np.abs(audio_chunk)) + 1e-8)  # Avoid division by zero

        fft_result = np.fft.fft(audio_chunk, n=int(sample_rate * sync_duration))
        freqs = np.fft.fftfreq(len(audio_chunk), 1 / sample_rate)
        fft_magnitude = np.abs(fft_result)

        # Find the dominant frequency
        dominant_freq_index = np.argmax(fft_magnitude)
        dominant_freq = abs(freqs[dominant_freq_index])

        # Check if the dominant frequency matches the sync frequency
        if abs(dominant_freq - sync_frequency) < tolerance:
            return True
        return False
    
    def find_sync_start(self, audio: np.ndarray, sync_duration: float = sync_duration) -> int:
        """Locate the start of the synchronization tone."""
        sync_length = int(sync_duration * sample_rate)
        for i in range(0, len(audio) - sync_length+1, sync_length//4):# Overlap chunks by 75%
            chunk = audio[i:i + sync_length]
            if self.detect_sync_tone(chunk):
                print(f"Sync tone detected at index {i}")
                return i + sync_length  # Start decoding after the sync tone
        print("Sync tone not detected in the audio")
        return -1

    def __reform_bit_String(self, audio, frames_per_bit: int, freq_tolerance=150) -> str:
        """Reform a bit string from the audio samples."""
        bit_string = ""
        fft_length = 2 ** int(np.ceil(np.log2(frames_per_bit)))

        for i in range(0, len(audio), frames_per_bit):
            chunk = audio[i:i+frames_per_bit]
            if len(chunk) < frames_per_bit:
                break
            if len(chunk) < fft_length:
                chunk = np.pad(chunk, (0, fft_length - len(chunk)), mode='constant')
            
            fft_result = np.fft.fft(chunk, n=fft_length)
            freqs = np.fft.fftfreq(len(chunk), 1 / sample_rate)
            fft_magnitude = np.abs(fft_result)
            
            dominant_freq_index = np.argmax(fft_magnitude)
            dominant_freq = abs(freqs[dominant_freq_index])

            # Determine bit by closest frequency
            if abs(dominant_freq - 1200) < freq_tolerance:
                bit_string += "0"
            elif abs(dominant_freq - 2200) < freq_tolerance:
                bit_string += "1"
        return bit_string
    
    def read_from_file(self, filename:str) -> str:
        sr, audio_data = read(filename)
        if len(audio_data.shape) > 1:
            audio_data = audio_data[:, 0]# Use only one channel for decoding

        baud_rate = self.baud_rate
        frames_per_bit = self.frames_per_bit
        print("Frames per bit:", frames_per_bit)

        filtered_audio = self.bandpass_filter(audio_data, 1000, 2500, sr)

        bit_string = self.__reform_bit_String(filtered_audio, frames_per_bit)

        try:
            original_message = self.__byte_converter.bitStrToBytes(bit_string).decode('utf-8', errors='ignore')
            return original_message
        except UnicodeDecodeError:
            print("Error decoding message.")
            return ""

    def receive_with_microphone_realtime(self, timeout: float, save_to_file: bool = False, output_filename: str = "received_audioTest.wav") -> str:
        print("Listening for signal...")

        audio_chunks = []
        recorded_frames = 0
        max_frames = int(timeout * sample_rate)
        frames_per_bit = sample_rate // self.baud_rate
        chunk_size = int(sample_rate * 0.1)  # Process in 100 ms chunks

        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            input=True,
            frames_per_buffer=chunk_size
        )

        try:
            while recorded_frames < max_frames:
                # Read a chunk of audio data
                audio_chunk = np.frombuffer(stream.read(chunk_size, exception_on_overflow=False), dtype=np.int16)
                recorded_frames += len(audio_chunk)

                if save_to_file:
                    audio_chunks.append(audio_chunk)

                # Detect synchronization tone (before bandpass filtering)
                sync_start = self.find_sync_start(audio_chunk)
                if sync_start != -1:
                    print("Synchronization tone detected. Decoding message...")

                    # Remove the sync tone and start decoding the remaining signal
                    remaining_audio = audio_chunk[sync_start:]

                    # Bandpass filter the audio to isolate the AFSK tones (1200-2200 Hz)
                    filtered_audio = self.bandpass_filter(remaining_audio, 1000, 2500, sample_rate)

                    # Decode the filtered audio
                    bit_string = self.__reform_bit_String(filtered_audio, frames_per_bit)

                    try:
                        # Convert the bit string to the original message
                        decoded_message = self.__byte_converter.bitStrToBytes(bit_string).decode('utf-8', errors='ignore')
                        return decoded_message
                    except UnicodeDecodeError:
                        print("Error decoding message. Retrying...")
                        continue

        finally:
            # Cleanup the stream
            stream.stop_stream()
            stream.close()
            p.terminate()

            # Save the recorded audio to a file if enabled
            if save_to_file and audio_chunks:
                print(f"Saving recorded audio to {output_filename}...")
                full_audio = np.concatenate(audio_chunks)
                with wave.open(output_filename, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
                    wf.setframerate(sample_rate)
                    wf.writeframes(full_audio.tobytes())

        print("Timeout: No signal detected.")
        return ""


# test_msg = "Hello, this is a test message. "*2
# sender = Sender()
# sender.send_msg(test_msg)
# sender.write_to_file(test_msg, "outputAudioTest.wav")

receiver = Receiver()
# original_msg_file = receiver.read_from_file("outputAudioTest.wav")
# print("original message from file:", original_msg_file)

original_msg = receiver.receive_with_microphone_realtime(6.0, save_to_file=True)
print("message received with microphone:", original_msg)
msg = receiver.read_from_file("received_audioTest.wav")
print(msg)

