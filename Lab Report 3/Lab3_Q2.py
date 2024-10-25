import numpy as np
import matplotlib.pyplot as plt

# Change Font Size for Readability
plt.rcParams.update({'font.size': 14})
plt.figure(figsize=(10, 8))



'''Part A: Plot the original stock data'''
# Extract col 1 from sp500.csv
# Skip 2 rows to prevent odd number issue and skip header
openingValue = np.loadtxt('sp500.csv', dtype=float, usecols=1, 
                  skiprows=2, delimiter=',')

# Each datapoint corresponds to one business day
businessDays = np.arange(len(openingValue))


plt.plot(businessDays, openingValue)
plt.xlabel("Number of Business Days")
plt.ylabel("Opening Value")
plt.title("S&P Stock Index Opening Value (2014-19)")
plt.savefig("Lab3Q2a.png")
# plt.show()
plt.clf()



'''Part B: fft+ifft equivalent to original'''

# apply rfft and irfft in sequence
openingValue_fft = np.fft.rfft(openingValue)
openingValue2 = np.fft.irfft(openingValue_fft)

# plot the difference between orignal data and double transformed data
plt.plot(businessDays, openingValue-openingValue2)
plt.xlabel("Number of Business Days")
plt.ylabel("Difference in Opening Value")
plt.title("Difference between Original and Transformed Data")
plt.savefig("Lab3Q2b.png")
# plt.show()
plt.clf()


# Test earlier when the number of data was 1259
print(f"Length of original Data: {len(openingValue)}")
print(f"Length of transformed Data: {len(openingValue2)}")





"""Part C: Smooth the Data"""
""" Remove any variations less than 6 months"""

# Same as class: Does np.arange(0, len(businessDays)//2+1) /len(businessDays)
frequencies = np.fft.rfftfreq(len(openingValue))  # 1/days
sixMonthFrequency = 1/120  # frequency of 120 days

# Set all frequencies higher than sixMonthFrequency to 0
openingValue_fft[frequencies>sixMonthFrequency] = 0.


openingValue_filtered = np.fft.irfft(openingValue_fft)


# Plot the original and filtered data overlayed
plt.plot(businessDays, openingValue, label='Original', alpha=0.6)
plt.plot(businessDays, openingValue_filtered, label='Filtered', color='k')
plt.xlabel("Number of Business Days")
plt.ylabel("Opening Value")
plt.title("Original vs Filtered S&P500 Index")
plt.legend()
plt.savefig("Lab3Q2c")
# plt.show()
plt.clf()

