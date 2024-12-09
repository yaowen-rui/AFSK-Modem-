import matplotlib.pyplot as plt
from scipy.io.wavfile import write, read
import numpy as np

# plot --- synchronization with sync tone 

# use the received recored audio via mic
#Plot the initial portion of the audio (first 10,000 samples) to confirm the presence of the sync tone:
sample_rate, audio = read("received_audioTest.wav")
plt.figure(figsize=(12, 6))
plt.plot(audio[:10000], label="Recorded Audio")
plt.title("Recorded Audio Signal")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

sync_duration = 3.0
sample_rate = 48000
sync_length = int(sync_duration * sample_rate)
chunk = audio[:sync_length]  # Extract the suspected sync tone region
fft_result = np.fft.fft(chunk)
freqs = np.fft.fftfreq(len(chunk), 1 / sample_rate)
fft_magnitude = np.abs(fft_result)

plt.figure(figsize=(10, 6))
plt.plot(freqs[:len(freqs)//2], fft_magnitude[:len(freqs)//2], label="Frequency Spectrum")
plt.title("Frequency Spectrum of Sync Tone Region")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.legend()
plt.show()
# The second plot shows a dominant peak at 3000 Hz,
#  which matches the expected sync tone frequency. 
# This confirms that the sync tone is being captured in the recorded audio.
