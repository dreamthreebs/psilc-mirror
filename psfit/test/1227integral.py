import numpy as np
from scipy.integrate import quad

# Constants
sigma = np.deg2rad(63) /  60 / np.sqrt(8 * np.log(2))  # example value for sigma
pi = np.pi

# Function to be integrated
def integrand(theta, sigma):
    return 1/(2*pi*sigma**2) * np.exp(-theta**2/(2*sigma**2)) * np.sin(theta)

# Numerical integration
integral, error = quad(integrand, 0, pi, args=(sigma,))


print(f'{integral=}')
print(f'{integral*np.pi*2=}')

