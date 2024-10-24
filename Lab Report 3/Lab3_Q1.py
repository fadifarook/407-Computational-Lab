from scipy.io.wavfile import read, write
from numpy import empty
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams.update({'font.size': 14})


# Read the data into two stereo channels, labelled 0 and 1
# sample is the sampling rate (44100 Hz)
# data is the data in each channel, dimensions [2, nsamples]
sample, data = read('407-Computational-Lab/Lab Report 3/GraviteaTime.wav')  # GraviteaTime.wav
channel_0 = data[:, 0]
channel_1 = data[:, 1]
nsamples = len(channel_0)  # number of values

time_array = np.arange(0, nsamples/sample, 1/sample)

print(sample, nsamples)

# Group 1: Time vs Amplitude for Both Channels
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(time_array, channel_0, label='Channel 0')
plt.title('Time vs Amplitude - Channel 0')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(time_array, channel_1, label='Channel 1', color='orange')
plt.title('Time vs Amplitude - Channel 1')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.tight_layout()
plt.savefig("Lab3Q1a.png")
plt.show()

# Fourier Transform calculations
channel_0_fft, channel_1_fft = np.fft.rfft(channel_0), np.fft.rfft(channel_1)
frequency_array = np.fft.rfftfreq(len(time_array), 1/sample)

# Apply filtering: zero frequencies above 880 Hz
channel_0_fft[frequency_array > 880] = 0.0
channel_1_fft[frequency_array > 880] = 0.0

# Group 2: 2x2 Fourier Transform and Filtered Fourier Transform
plt.figure(figsize=(12, 8))

# Channel 0 original Fourier
plt.subplot(2, 2, 1)
plt.plot(frequency_array, np.abs(np.fft.rfft(channel_0)), label='Original FFT')
plt.title('Channel 0 - Original FFT')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.xlim(0, 1000)
plt.axvline(880, color='red', linestyle='--')

# Channel 1 original Fourier
plt.subplot(2, 2, 2)
plt.plot(frequency_array, np.abs(np.fft.rfft(channel_1)), label='Original FFT')
plt.title('Channel 1 - Original FFT')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.xlim(0, 1000)
plt.axvline(880, color='red', linestyle='--')

# Channel 0 filtered Fourier
plt.subplot(2, 2, 3)
plt.plot(frequency_array, np.abs(channel_0_fft), label='Filtered FFT', color='green')
plt.title('Channel 0 - Filtered FFT')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.xlim(0, 1000)
plt.axvline(880, color='red', linestyle='--')

# Channel 1 filtered Fourier
plt.subplot(2, 2, 4)
plt.plot(frequency_array, np.abs(channel_1_fft), label='Filtered FFT', color='green')
plt.title('Channel 1 - Filtered FFT')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.xlim(0, 1000)
plt.axvline(880, color='red', linestyle='--')

plt.tight_layout()
plt.savefig("Lab3Q1b.png")
plt.show()



# Transform back to time domain
channel_0_filtered = np.fft.irfft(channel_0_fft)
channel_1_filtered = np.fft.irfft(channel_1_fft)

# Group 3: 2x2 Original and Filtered Inverse
plt.figure(figsize=(12, 8))

# Channel 0 original signal
plt.subplot(2, 2, 1)
plt.plot(time_array, channel_0)
plt.title('Channel 0 - Original Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# Channel 1 original signal
plt.subplot(2, 2, 2)
plt.plot(time_array, channel_1)
plt.title('Channel 1 - Original Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# Channel 0 filtered signal
plt.subplot(2, 2, 3)
plt.plot(time_array, channel_0_filtered, color='green')
plt.title('Channel 0 - Filtered Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# Channel 1 filtered signal
plt.subplot(2, 2, 4)
plt.plot(time_array, channel_1_filtered, color='green')
plt.title('Channel 1 - Filtered Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.savefig("Lab3Q1c.png")
plt.show()


# Group 4: Overlay of Original and Filtered Signals
plt.figure(figsize=(12, 6))

# Overlay for Channel 0
plt.subplot(2, 1, 1)
plt.plot(time_array, channel_0, label='Original', alpha=0.6)
plt.plot(time_array, channel_0_filtered, label='Filtered', linestyle='--', color='green')
plt.title('Channel 0 - Original vs Filtered')
plt.xlabel('Time (s)')
plt.xlim(0, 30e-3)
plt.ylabel('Amplitude')
plt.legend()

# Overlay for Channel 1
plt.subplot(2, 1, 2)
plt.plot(time_array, channel_1, label='Original', alpha=0.6)
plt.plot(time_array, channel_1_filtered, label='Filtered', linestyle='--', color='green')
plt.title('Channel 1 - Original vs Filtered')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.xlim(0, 30e-3)
plt.legend()

plt.tight_layout()
plt.savefig("Lab3Q1d.png")
plt.show()

# Output the data to a new .wav file
data_out = empty(data.shape, dtype=data.dtype)
data_out[:, 0] = channel_0_filtered
data_out[:, 1] = channel_1_filtered
write('output_file.wav', sample, data_out)
