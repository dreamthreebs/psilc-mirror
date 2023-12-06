import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Legendre
from scipy.interpolate import interp1d

def chebyshev_nodes(n, a=-1, b=1):
    return np.cos((2*np.arange(1, n+1) - 1) / (2*n) * np.pi) * (b - a) / 2 + (b + a) / 2

def interpolate_legendre(l, num_nodes):
    # Generate Chebyshev nodes
    nodes = chebyshev_nodes(num_nodes)

    # Evaluate the Legendre polynomial at these nodes
    leg_vals = Legendre.basis(l)(nodes)

    # Use cubic spline interpolation
    spline = interp1d(nodes, leg_vals, kind='cubic', bounds_error=False, fill_value="extrapolate")

    return spline

# Degree of the Legendre polynomial
l = 100

# Number of nodes for interpolation
num_nodes = 200

# Create the interpolation function
interp_func = interpolate_legendre(l, num_nodes)

# Plotting
x = np.linspace(-1, 1, 1000)
y = interp_func(x)
plt.plot(x, y, label='Interpolated')
plt.plot(x, Legendre.basis(l)(x), label='Actual', linestyle='dashed')
plt.legend()
plt.show()

