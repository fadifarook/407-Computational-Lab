import numpy as np
import matplotlib.pyplot as plt




from numpy import loadtxt
SLP = loadtxt('407-Computational-Lab/Lab Report 3/SLP.txt')
Longitude = loadtxt('407-Computational-Lab/Lab Report 3/lon.txt')
Times = loadtxt('407-Computational-Lab/Lab Report 3/times.txt')



print(len(Times), len(Longitude))

print(SLP.shape)


print(Longitude[20] -Longitude[19])




plt.contourf(Longitude, Times, SLP)
plt.colorbar()
plt.show()
plt.clf()



"""Do a 1D fourier transform in the longitudinal space"""
SLP_fft = np.fft.rfft(SLP, axis=1)
print(SLP_fft.shape)

wavenumbers = np.fft.rfftfreq(len(Longitude), Longitude[1]-Longitude[0])
print(len(wavenumbers))

SLP_m3 = SLP_fft*0
SLP_m5 = SLP_fft*0

temp3 = SLP_fft[:, 3]  # time comes back
temp5 = SLP_fft[:, 5]

for i in range(73):
    SLP_m3[:, 3] = temp3
    SLP_m5[:, 5] = temp5

print(SLP_m3.shape)

# wavenumbers /= Longitude


plt.contourf(wavenumbers, Times, np.abs(SLP_m3))   # Plot ABS
# plt.show()
plt.clf()






""" Filter and Return"""



SLP_fltered3 = np.fft.irfft(SLP_m3, axis=1)
SLP_fltered5 = np.fft.irfft(SLP_m5, axis=1)

# print(SLP_fltered.shape)

plt.contourf(Longitude, Times, SLP_fltered5)
plt.colorbar()
plt.show()