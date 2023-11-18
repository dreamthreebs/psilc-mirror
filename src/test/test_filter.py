import numpy as np
import matplotlib.pyplot as plt

def gaussian_low_pass(frequencies, cutoff):
    """ Gaussian low-pass filter function. """
    return np.exp(- (frequencies / cutoff) ** 2)

# Define the range for x
x = np.linspace(-1, 1, 400)

# Create a delta function centered at x=0
delta = np.zeros_like(x)
delta[len(x)//2] = 1  # Setting the central value to be high

# Perform Fourier transform of the delta function
ft_delta = np.fft.fftshift(np.fft.fft(delta))

# Frequency array (normalizing to the length of x for proper scaling)
freq = np.fft.fftshift(np.fft.fftfreq(len(x), d=x[1]-x[0]))

# Define different cutoff frequencies for the Gaussian low-pass filter
original_cutoff_frequency = 10  # Original cutoff frequency
larger_cutoff_frequency = 20    # Larger cutoff frequency
smaller_cutoff_frequency = 5    # Smaller cutoff frequency

# Create Gaussian low-pass filters with the defined cutoff frequencies
gaussian_filter_original = gaussian_low_pass(freq, original_cutoff_frequency)
gaussian_filter_larger = gaussian_low_pass(freq, larger_cutoff_frequency)
gaussian_filter_smaller = gaussian_low_pass(freq, smaller_cutoff_frequency)

# Apply these Gaussian filters to the Fourier transform of the delta function
gaussian_filtered_ft_delta_original = ft_delta * gaussian_filter_original
gaussian_filtered_ft_delta_larger = ft_delta * gaussian_filter_larger
gaussian_filtered_ft_delta_smaller = ft_delta * gaussian_filter_smaller

# Perform the inverse Fourier transform to get back to real space
gaussian_filtered_delta_original = np.fft.ifft(np.fft.ifftshift(gaussian_filtered_ft_delta_original))
gaussian_filtered_delta_larger = np.fft.ifft(np.fft.ifftshift(gaussian_filtered_ft_delta_larger))
gaussian_filtered_delta_smaller = np.fft.ifft(np.fft.ifftshift(gaussian_filtered_ft_delta_smaller))

# Plotting the results with Gaussian low-pass filters
fig, axes = plt.subplots(4, 1, figsize=(10, 10))

# Plot the filtered signal with the original Gaussian low-pass filter
axes[0].plot(x, gaussian_filtered_delta_original.real)
axes[0].set_title("Gaussian Filtered Delta Function with Original Cutoff Frequency")
axes[0].set_xlim([-1, 1])

# Plot the Gaussian low-pass filter in Fourier space
axes[1].plot(freq, gaussian_filter_original)
axes[1].set_title("Gaussian Low-Pass Filter in Fourier Space")
axes[1].set_xlim([-50, 50])

# Plot the filtered signal with a larger Gaussian low-pass filter
axes[2].plot(x, gaussian_filtered_delta_larger.real)
axes[2].set_title("Gaussian Filtered Delta Function with Larger Cutoff Frequency")
axes[2].set_xlim([-1, 1])

# Plot the filtered signal with a smaller Gaussian low-pass filter
axes[3].plot(x, gaussian_filtered_delta_smaller.real)
axes[3].set_title("Gaussian Filtered Delta Function with Smaller Cutoff Frequency")
axes[3].set_xlim([-1, 1])

plt.tight_layout()
plt.show()

