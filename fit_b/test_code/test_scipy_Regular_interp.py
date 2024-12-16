import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

# Define the 2D grid
x = np.linspace(0, 4, 10)  # Grid points for the x-axis
y = np.linspace(0, 4, 10)  # Grid points for the y-axis
X, Y = np.meshgrid(x, y, indexing='ij')  # Create a 2D grid

# Define a 2D function over the grid
Z = np.sin(X) * np.cos(Y)

# Create the interpolator
interpolator = RegularGridInterpolator((x, y), Z)

# Points to interpolate
x_new = np.linspace(0, 4, 50)  # Fine grid for x-axis
y_new = np.linspace(0, 4, 50)  # Fine grid for y-axis
X_new, Y_new = np.meshgrid(x_new, y_new, indexing='ij')
points = np.array([X_new.ravel(), Y_new.ravel()]).T  # Create points for interpolation

# Perform interpolation
Z_new = interpolator(points).reshape(X_new.shape)

# Visualize the results
plt.figure(figsize=(12, 6))

# Original grid
plt.subplot(1, 2, 1)
plt.contourf(X, Y, Z, levels=20, cmap='viridis')
plt.title('Original Grid')
plt.colorbar(label='Value')

# Interpolated grid
plt.subplot(1, 2, 2)
plt.contourf(X_new, Y_new, Z_new, levels=50, cmap='viridis')
plt.title('Interpolated Grid')
plt.colorbar(label='Value')

plt.tight_layout()
plt.show()

