from scipy.io.wavfile import read, write
from numpy import empty
import numpy as np
import matplotlib.pyplot as plt


# read the data into two stereo channels, labelled 0 and 1
# sample is the sampling rate (44100 Hz)
# data is the data in each channel, dimensions [2, nsamples]
sample, data = read('407-Computational-Lab/Lab Report 3/GraviteaTime.wav') # GraviteaTime.wav
channel_0 = data[:, 0]
channel_1 = data[:, 1]
nsamples = len(channel_0)  # number of values


time_array = np.arange(0, nsamples/sample , 1/sample)

print(sample, nsamples)



plt.plot(time_array, channel_0)
plt.plot(time_array, channel_1)
# plt.show()
plt.clf()





""" Fourier Transform PLOT ABSOLUTE """

channel_0_fft, channel_1_fft = np.fft.rfft(channel_0), np.fft.rfft(channel_1)
frequency_array = np.fft.rfftfreq(len(time_array), 1/sample)

channel_0_fft[frequency_array>880], channel_1_fft[frequency_array>880] = 0., 0.


plt.plot(frequency_array, np.abs(channel_0_fft))
plt.plot(frequency_array, np.abs(channel_1_fft))
plt.axvline(880)
plt.xlim(0, 1000)
# plt.show()
plt.clf()





"""Transoform Back, plot"""



channel_0_filtered, channel_1_filtered = np.fft.irfft(channel_0_fft), np.fft.irfft(channel_1_fft)


plt.plot(time_array, channel_0)
plt.plot(time_array, channel_0_filtered)
plt.xlim(0, 30e-3)
plt.show()
plt.clf()


plt.plot(time_array, channel_1)
plt.plot(time_array, channel_1_filtered)
plt.xlim(0, 30e-3)
plt.show()





















"""Output the Data"""
# # ... work with the data to create new arrays channel_0_out and channel_1_out,
# # each of length nsamples, containing values convertible to int16
# # create & fill empty output array data_out with the same shape and datatype as "data"



data_out = empty(data.shape, dtype = data.dtype)
data_out[:, 0] = channel_0_filtered
data_out[:, 1] = channel_1_filtered
# write the output array to a new .wav file
write('output_file.wav', sample, data_out)




