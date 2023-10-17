import numpy as np
import matplotlib.pyplot as plt

# Define N and k
N = 1000
k = np.arange(N)

# Create X[k] where X[k] = 1 for k in [0, 10] and 0 otherwise
X_k = np.where(k <= 10, 1, 0)

# Perform inverse DFT to get x[n]
x_n = np.fft.ifft(X_k) * N

# Truncate x[n] to [0, 500]
x_n_truncated = x_n[:501]

# Perform forward DFT on truncated x[n]
X_k_transformed = np.fft.fft(x_n_truncated) / 501

# Plot the real part of X[k] for k in [0, 10]
plt.figure()
plt.plot(k[:11], np.real(X_k_transformed)[:11])
plt.xlabel('k')
plt.ylabel('Re(X[k])')
plt.title('Forward DFT on Truncated x[n]: Re(X[k]) for k in [0, 10]')
plt.show()

