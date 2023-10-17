import numpy as np
import matplotlib.pyplot as plt

# Define N for inverse DFT and k
N_inv = 10000
k = np.arange(N_inv)

# Create X[k] where X[k] = 1 for k in [1, 10] and 0 otherwise
X_k = np.where((k >= 1) & (k <= 100), 1, 0)

# Perform inverse DFT to get x[n]
x_n = np.fft.ifft(X_k) * N_inv

# Plot the real part of x[n]
plt.figure()
plt.plot(np.real(x_n))
plt.xlabel('n')
plt.ylabel('Re(x[n])')
plt.title('Inverse DFT: Re(x[n]) with N=1000')
plt.show()

# Truncate x[n] for limited N in DFT
N_dft = 3000
x_n_truncated = x_n[3000:6000]

# Perform forward DFT to get back X[k]
X_k_transformed = np.fft.fft(x_n_truncated) / N_dft

# Calculate the power spectrum
power_spectrum = np.abs(X_k_transformed)**2

# Plot the power spectrum for k in [0, 20]
plt.figure()
plt.plot(power_spectrum[:200])
plt.xlabel('k')
plt.ylabel('Power')
plt.title('Power Spectrum with N=500 in DFT')
plt.show()
# Apply Hamming window to truncated x[n]
hamming_window = np.hamming(N_dft)
x_n_windowed = x_n_truncated * hamming_window

# Perform forward DFT on windowed x[n]
X_k_windowed = np.fft.fft(x_n_windowed) / N_dft

# Calculate the power spectrum
power_spectrum_windowed = np.abs(X_k_windowed)**2

# Plot the power spectrum for k in [0, 20]
plt.figure()
plt.plot(power_spectrum_windowed[:21])
plt.xlabel('k')
plt.ylabel('Power')
plt.title('Power Spectrum with Hamming Window and N=500 in DFT')
plt.show()

