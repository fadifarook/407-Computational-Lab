import numpy as np
import matplotlib.pyplot as plt




from numpy import loadtxt
SLP = loadtxt('407-Computational-Lab/Lab Report 3/SLP.txt')
Longitude = loadtxt('407-Computational-Lab/Lab Report 3/lon.txt')
Times = loadtxt('407-Computational-Lab/Lab Report 3/times.txt')



print(len(Times), len(Longitude))

print(SLP.shape)


print(Longitude[20] -Longitude[19])




plt.contourf(Longitude, Times, SLP, cmap='RdBu')
plt.colorbar()
plt.savefig("Lab3Q3a.png")
plt.show()
plt.clf()



"""Do a 1D fourier transform in the longitudinal space"""
SLP_fft = np.fft.rfft(SLP, axis=1)
print(SLP_fft.shape)


wavenumbers = np.fft.rfftfreq(len(Longitude), Longitude[1]-Longitude[0])
print(len(wavenumbers))

wavenumbers = np.arange(73)

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
plt.xlim(0, 6)
# plt.show()
plt.clf()






""" Filter and Return"""



SLP_fltered3 = np.fft.irfft(SLP_m3, axis=1)
SLP_fltered5 = np.fft.irfft(SLP_m5, axis=1)

# print(SLP_fltered.shape)

# Create a figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# First subplot for SLP_fltered3
contour1 = axs[0].contourf(Longitude, Times, SLP_fltered3, cmap='RdBu')
axs[0].set_title('Filtered SLP - m = 3')
axs[0].set_xlabel('Longitude')
axs[0].set_ylabel('Time')
fig.colorbar(contour1, ax=axs[0])

# Second subplot for SLP_fltered5
contour2 = axs[1].contourf(Longitude, Times, SLP_fltered5, cmap='RdBu')
axs[1].set_title('Filtered SLP - m = 5')
axs[1].set_xlabel('Longitude')
fig.colorbar(contour2, ax=axs[1])

# Show the plot
plt.tight_layout()
plt.savefig("Lab3Q3b.png")
plt.show()
