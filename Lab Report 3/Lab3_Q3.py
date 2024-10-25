import numpy as np
import matplotlib.pyplot as plt

"""Code for Q3: Plots Sea Level Pressure based on longitude and time
Extracts wavenumber 3 and 5 in fourier space and plots then in longitude space"""


# Change Font Size for Readability
plt.rcParams.update({"font.size": 14})

# Load all the data
SLP = np.loadtxt("SLP.txt")
Longitude = np.loadtxt("lon.txt")
Times = np.loadtxt("times.txt")


# Original Plot to visualize SLP
plt.contourf(Longitude, Times, SLP, cmap="RdBu")
plt.colorbar(label="Pressure [hPa]")
plt.xlabel("Longitude [degrees]")
plt.ylabel("Time [days]")
plt.title("Deviation of SLP from Mean at $50^{\\circ}$ Latitude")
plt.savefig("Lab3Q3a.png")
# plt.show()
plt.clf()


""" Part A: Extract m=3 and m=5"""

"""Do a 1D fourier transform in the longitudinal space"""
SLP_fft = np.fft.rfft(SLP, axis=1)

# Gets all the wavenumbers, only in the longitude fft
wavenumbers = np.arange(len(SLP_fft[1,]))


# Make two arrays, whose values are zero except
# at specific wavenumbers
SLP_fft_m5 = SLP_fft.copy()
SLP_fft_m3 = SLP_fft.copy()
SLP_fft_m3[:, wavenumbers != 3] = 0
SLP_fft_m5[:, wavenumbers != 5] = 0


# Return to longitude space
SLP_fltered3 = np.fft.irfft(SLP_fft_m3, axis=1)
SLP_fltered5 = np.fft.irfft(SLP_fft_m5, axis=1)


"""Plot both m=3 and m=5 plots"""
plt.figure(figsize=(14, 6))
plt.suptitle("Filtered Deviation of SLP from Mean at $50^{\\circ}$ Latitude")

# First subplot for SLP_fltered3
plt.subplot(1, 2, 1)
plt.contourf(Longitude, Times, SLP_fltered3, cmap="RdBu")
plt.title("Wavenumber = 3 Only")
plt.xlabel("Longitude [degrees]")
plt.ylabel("Time [days]")
plt.colorbar(label="Pressure [hPa]")

# Second subplot for SLP_fltered5
plt.subplot(1, 2, 2)
plt.contourf(Longitude, Times, SLP_fltered5, cmap="RdBu")
plt.title("Wavenumber = 5 Only")
plt.xlabel("Longitude [degrees]")
plt.ylabel("Time [days]")
plt.colorbar(label="Pressure [hPa]")


# Show the plot
plt.tight_layout()
plt.savefig("Lab3Q3b.png")
# plt.show()
plt.clf()
