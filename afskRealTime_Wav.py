import pyaudio
import numpy as np
import wave
import matplotlib.pyplot as plt
from scipy.io.wavfile import write, read
from scipy.signal import butter, sosfiltfilt, correlate
import time
import string
import threading

DEBUG = True  

def plot_signal(signal, title="Signal"):
    if not DEBUG:
        return
    plt.figure()
    plt.plot(signal)
    plt.title(title)
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()

def plot_fft(signal, sample_rate, title="FFT"):
    if not DEBUG:
        return
    n = len(signal)
    fft_result = np.fft.fft(signal)
    freqs = np.fft.fftfreq(n, 1/sample_rate)
    magnitude = np.abs(fft_result)
    plt.figure()
    plt.plot(freqs[:n//2], magnitude[:n//2])
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True)
    plt.show()

SAMPLE_RATE    = 48000
HIGH_AMPLITUDE = 32767
BAUD_RATE      = 300  
SYNC_FREQUENCY = 3000
SYNC_DURATION  = 3.0
PREAMBLE_TEXT  = "<<<PREAMBLE:>>>"
END_OF_MSG     = "<<<END>>>"

SPACE_CENTER = 1200
MARK_CENTER  = 2200
BAND_WIDTH   = 100

ALPHA        = 1.2
EPSILON      = 1e-8
EXTRA_MARGIN = 4


class AFSKWaves:
    @staticmethod
    def generate_space_tone(baud_rate: int):
        frames = SAMPLE_RATE // baud_rate
        t = np.linspace(0, 1/baud_rate, frames, endpoint=False)
        return (HIGH_AMPLITUDE * np.sin(2*np.pi * SPACE_CENTER * t)).astype(np.int16)

    @staticmethod
    def generate_mark_tone(baud_rate: int):
        frames = SAMPLE_RATE // baud_rate
        t = np.linspace(0, 1/baud_rate, frames, endpoint=False)
        return (HIGH_AMPLITUDE * np.sin(2*np.pi * MARK_CENTER * t)).astype(np.int16)

    @staticmethod
    def generate_sync_tone(baud_rate: int):
        t = np.linspace(0, SYNC_DURATION, int(SAMPLE_RATE * SYNC_DURATION), endpoint=False)
        return (HIGH_AMPLITUDE * np.sin(2*np.pi * SYNC_FREQUENCY * t)).astype(np.int16)


class ByteBitConverter:
    def bytes_to_bit_str(self, input_bytes: bytes) -> str:
        return ''.join(f'{byte:08b}' for byte in input_bytes)

    def bit_str_to_bytes(self, input_bits: str) -> bytes:
        result = []
        for i in range(0, len(input_bits), 8):
            chunk = input_bits[i:i+8]
            if len(chunk) < 8:
                chunk = chunk.ljust(8, '0')
            result.append(int(chunk, 2))
        return bytes(result)


class Sender:
    def __init__(self, baud_rate: int = BAUD_RATE):
        self.baud_rate   = baud_rate
        self.space_tone  = AFSKWaves.generate_space_tone(baud_rate)
        self.mark_tone   = AFSKWaves.generate_mark_tone(baud_rate)
        self.sync_tone   = AFSKWaves.generate_sync_tone(baud_rate)
        self.converter   = ByteBitConverter()

    def prepare_for_sending(self, message: str):
        full_message = PREAMBLE_TEXT + message + END_OF_MSG
        data = full_message.encode('utf-8')
        bit_string = self.converter.bytes_to_bit_str(data)
        audio_data = []
        for bit in bit_string:
            if bit == '0':
                audio_data.extend(self.space_tone)
            else:
                audio_data.extend(self.mark_tone)
        message_audio = np.array(audio_data, dtype=np.int16)
        silence = np.zeros(SAMPLE_RATE, dtype=np.int16)
        return np.concatenate([self.sync_tone, message_audio, silence])

    def send_msg(self, message: str):
        import pyaudio
        if not message:
            return
        try:
            print(f"Transmitting: {message}")
            audio_samples = self.prepare_for_sending(message)
            pa = pyaudio.PyAudio()
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=SAMPLE_RATE,
                output=True,
                frames_per_buffer=256
            )
            stream.write(audio_samples.tobytes())
            print("Transmission complete.")
        except Exception as e:
            print("Error during transmission:", e)
        finally:
            try:
                stream.stop_stream()
            except:
                pass
            stream.close()
            pa.terminate()

    def write_to_file(self, message: str, filename: str):
        audio_samples = self.prepare_for_sending(message)
        write(filename, SAMPLE_RATE, audio_samples)


class Receiver:
    def __init__(self, baud_rate: int = BAUD_RATE):
        self.baud_rate      = baud_rate
        self.frames_per_bit = SAMPLE_RATE // baud_rate
        self.converter      = ByteBitConverter()
        self.noise_profile  = None

    def bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        nyquist = 0.5*fs
        low     = lowcut / nyquist
        high    = highcut / nyquist
        sos     = butter(order, [low, high], btype='band', output='sos')
        return sosfiltfilt(sos, data)

    def measure_noise_profile(self, duration=1.0, chunk_size=256):
        import pyaudio
        pa = pyaudio.PyAudio()
        try:
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=chunk_size
            )
            needed = int(duration*SAMPLE_RATE)
            frames = []
            collected = 0
            while collected < needed:
                data = stream.read(chunk_size, exception_on_overflow=False)
                block = np.frombuffer(data, dtype=np.int16)
                frames.append(block)
                collected += len(block)
            noise_data = np.concatenate(frames)
        except:
            noise_data = np.zeros(int(duration*SAMPLE_RATE))
        finally:
            try:
                stream.stop_stream()
            except:
                pass
            stream.close()
            pa.terminate()

        n_fft = 1024
        all_spectra = []
        step = n_fft
        for i in range(0, len(noise_data), step):
            chunk = noise_data[i:i+step]
            if len(chunk) < step:
                break
            window = np.hamming(len(chunk))
            fft_result = np.fft.fft(chunk*window, n=n_fft)
            magnitude = np.abs(fft_result)
            all_spectra.append(magnitude)

        if len(all_spectra) > 0:
            avg_spectrum = np.mean(all_spectra, axis=0)
        else:
            avg_spectrum = np.zeros(n_fft)
        self.noise_profile = avg_spectrum

    def spectral_subtraction(self, fft_magnitude):
        if self.noise_profile is None:
            return fft_magnitude
        clean = fft_magnitude - ALPHA*self.noise_profile
        return np.maximum(clean, 0)

    def compute_band_power(self, fft_mag, freqs, center, bandwidth):
        lower = center - bandwidth
        upper = center + bandwidth
        mask  = (freqs >= lower) & (freqs <= upper)
        return np.sum(fft_mag[mask])

    def detect_bit_adaptive(self, chunk):
        n_fft = 1024
        window = np.hamming(len(chunk))
        fft_result = np.fft.fft(chunk*window, n=n_fft)
        fft_magnitude = np.abs(fft_result)
        clean_mag = self.spectral_subtraction(fft_magnitude)
        freqs = np.fft.fftfreq(n_fft, 1/SAMPLE_RATE)
        freqs_abs = np.abs(freqs)
        space_pwr = self.compute_band_power(clean_mag, freqs_abs, SPACE_CENTER, BAND_WIDTH)
        mark_pwr  = self.compute_band_power(clean_mag, freqs_abs, MARK_CENTER,  BAND_WIDTH)
        return "0" if space_pwr > mark_pwr else "1"

    def _reform_bit_string_adaptive(self, audio):
        bit_string = ""
        for i in range(0, len(audio), self.frames_per_bit):
            chunk = audio[i:i+self.frames_per_bit]
            if len(chunk) < self.frames_per_bit:
                break
            bit = self.detect_bit_adaptive(chunk)
            bit_string += bit
        return bit_string

    def decode_with_offset(self, audio, offset):
        bit_string = ""
        for i in range(offset, len(audio), self.frames_per_bit):
            chunk = audio[i:i+self.frames_per_bit]
            if len(chunk) < self.frames_per_bit:
                break
            bit = self.detect_bit_adaptive(chunk)
            bit_string += bit
        return bit_string

    def score_text(self, text):
        valid_chars = set(string.ascii_letters+string.digits+" ,.-;:!?'\"")
        if not text:
            return 0
        return sum(1 for c in text if c in valid_chars)/len(text)

    def find_preamble_offset(self, bit_string, preamble_bits):
        arr_bits = np.array([int(b) for b in bit_string], dtype=float)
        arr_pre  = np.array([int(b) for b in preamble_bits], dtype=float)
        pre_norm = np.linalg.norm(arr_pre)
        if pre_norm == 0:
            pre_norm = 1
        length_diff = len(arr_bits) - len(arr_pre) + 1
        if length_diff <= 0:
            return -1,0
        scores = []
        for i in range(length_diff):
            segment = arr_bits[i:i+len(arr_pre)]
            seg_norm = np.linalg.norm(segment)
            if seg_norm == 0:
                norm_corr = 0
            else:
                norm_corr = np.dot(segment, arr_pre)/(seg_norm*pre_norm)
            scores.append(norm_corr)
        scores = np.array(scores)
        if len(scores) == 0:
            return -1,0
        best_idx  = int(np.argmax(scores))
        best_score= scores[best_idx]
        return best_idx, best_score

    def align_and_decode(self, audio)->(str,str):
        best_score = 0
        best_text  = ""
        best_bits  = ""
        for offset in range(self.frames_per_bit):
            candidate_bits = self.decode_with_offset(audio, offset)
            try:
                candidate_text = ByteBitConverter().bit_str_to_bytes(candidate_bits).decode('utf-8', errors='ignore')
            except:
                candidate_text = ""
            score = self.score_text(candidate_text)
            if score > best_score:
                best_score = score
                best_text  = candidate_text
                best_bits  = candidate_bits
        return best_text, best_bits

    def stop_at_end_marker(self, text: str) -> str:
        idx = text.find(END_OF_MSG)
        return text[:idx] if idx != -1 else text

    def offline_decode(self, audio_data: np.ndarray):
        filtered = self.bandpass_filter(audio_data, 1000, 2500, SAMPLE_RATE)
        # Show wave/FFT for mic
        plot_signal(filtered, "Filtered Signal (Mic)")
        plot_fft(filtered, SAMPLE_RATE, "FFT of Filtered Signal (Mic)")

        sync_samples = AFSKWaves.generate_sync_tone(BAUD_RATE)
        if len(filtered) < len(sync_samples):
            return ""
        corr = correlate(filtered, sync_samples, mode='valid')
        peak_idx = np.argmax(corr)

        tail_preserve = len(sync_samples) + (EXTRA_MARGIN*self.frames_per_bit)
        start_idx = max(0, peak_idx+len(sync_samples)-tail_preserve)
        msg_signal = filtered[start_idx:]

        best_text, best_bits = self.align_and_decode(msg_signal)

        preamble_bits = ByteBitConverter().bytes_to_bit_str(PREAMBLE_TEXT.encode('utf-8'))
        idx,score = self.find_preamble_offset(best_bits, preamble_bits)
        final_text = best_text
        if score >= 0.6 and idx != -1:
            tail_str = best_bits[idx+len(preamble_bits):]
            try:
                final_text = ByteBitConverter().bit_str_to_bytes(tail_str).decode('utf-8', errors='ignore')
            except:
                pass
        final_text = self.stop_at_end_marker(final_text)
        return final_text

    def record_fixed_time(self, record_secs=8, chunk_size=256):
        import pyaudio
        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=chunk_size
        )
        frames=[]
        total_needed = record_secs * SAMPLE_RATE
        collected = 0
        while collected < total_needed:
            data = stream.read(chunk_size, exception_on_overflow=False)
            block = np.frombuffer(data, dtype=np.int16)
            frames.append(block)
            collected += len(block)
        try:
            stream.stop_stream()
        except:
            pass
        stream.close()
        pa.terminate()
        return np.concatenate(frames)

    def read_from_file(self, filename:str)->str:
        sr,audio_data = read(filename)
        if len(audio_data.shape)>1:
            audio_data = audio_data[:,0]
        # Plots for file
        plot_signal(audio_data, "Raw signal from file")
        plot_fft(audio_data, sr, "FFT of raw signal (File)")

        filtered = self.bandpass_filter(audio_data,1000,2500,sr)
        plot_signal(filtered, "Filtered Signal (File)")
        plot_fft(filtered, sr, "FFT of Filtered Signal (File)")

        sync_len = int(SAMPLE_RATE * SYNC_DURATION)
        filtered = filtered[sync_len:]
        best_text,best_bits = self.align_and_decode(filtered)

        preamble_bits = ByteBitConverter().bytes_to_bit_str(PREAMBLE_TEXT.encode('utf-8'))
        idx,score     = self.find_preamble_offset(best_bits, preamble_bits)
        final_text    = best_text
        if score>=0.6 and idx!=-1:
            tail_str   = best_bits[idx+len(preamble_bits):]
            try:
                final_text = ByteBitConverter().bit_str_to_bytes(tail_str).decode('utf-8', errors='ignore')
            except:
                pass
        final_text = self.stop_at_end_marker(final_text)
        return final_text

if __name__=="__main__":
    receiver = Receiver()
    sender   = Sender()

    message  = "Hello, this is a test message."

    def record_task():
        receiver.measure_noise_profile(1.0, chunk_size=256)
        global recorded_data
        recorded_data = receiver.record_fixed_time(record_secs=8, chunk_size=256)

    rec_thread = threading.Thread(target=record_task)
    rec_thread.start()
    time.sleep(1)

    sender.send_msg(message)
    sender.write_to_file(message,"output.wav")

    rec_thread.join()

    print("Decoding from output.wav...")
    file_decoded = receiver.read_from_file("output.wav")
    print("Decoded from output.wav:", file_decoded)

    print("Decoding from mic...")
    real_decoded = receiver.offline_decode(recorded_data)
    print("Decoded from mic capture:", real_decoded)
