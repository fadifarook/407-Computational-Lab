import numpy as np
import matplotlib.pyplot as plt



openingValue = np.loadtxt('407-Computational-Lab/Lab Report 3/sp500.csv', dtype=float, usecols=1, 
                  skiprows=2, delimiter=',')

businessDays = np.arange(len(openingValue))


plt.plot(businessDays, openingValue)
plt.xlabel("Number of Business Days")
plt.ylabel("Opening Value")
plt.savefig("Lab3Q2a.png")
plt.show()



openingValue_fft = np.fft.rfft(openingValue)
openingValue2 = np.fft.irfft(openingValue_fft)

plt.plot(businessDays, openingValue-openingValue2)
plt.xlabel("Number of Business Days")
plt.ylabel("Opening Value")
plt.savefig("Lab3Q2b.png")
plt.show()
plt.clf()

print(len(openingValue), len(openingValue2))  # if i skip two rows it doesnt work, skip one row it works






""" Remove any variations less than 6 months"""

frequencies = np.fft.rfftfreq(len(openingValue), 1)

# plt.plot(frequencies, openingValue_fft)
# plt.show()


openingValue_fft[frequencies>0.008333] = 0.


openingValue_filtered = np.fft.irfft(openingValue_fft)


plt.plot(businessDays, openingValue)
plt.plot(businessDays, openingValue_filtered)
plt.xlabel("Number of Business Days")
plt.ylabel("Opening Value")
plt.title("Original vs Smoothed Data")
plt.savefig("Lab3Q2c")
plt.show()
plt.clf()

