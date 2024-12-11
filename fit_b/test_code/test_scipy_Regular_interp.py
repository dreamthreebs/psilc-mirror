import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

# Define the 1D grid and function values in reverse order
x = np.linspace(0, 10, 5)  # Grid points in descending order
y = np.sin(x)  # Function values at the grid points

# Reverse the order of grid and values to ascending order
x_sorted = x[::-1]
y_sorted = y[::-1]

# Create the interpolator
interpolator = RegularGridInterpolator((x_sorted,), y_sorted)

# Interpolate at new points
x_new = np.linspace(0, 10, 50)  # Points to interpolate
y_new = interpolator(x_new)

# Visualize the result
plt.plot(x, y, 'o', label="Original Points (Descending Order)")
plt.plot(x_new, y_new, '-', label="Interpolated Curve")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Interpolation with Data in Inverse Order")
plt.show()

