import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

beam = 63
sigma = np.deg2rad(beam) / 60 / (np.sqrt(8 * np.log(2)))
beam_size = np.deg2rad(beam) / 60
x = np.linspace(0, 3.5, 100)
theta = x * beam_size

def model(theta):
    return 1 / (2 * np.pi * sigma**2) * np.exp(- (theta)**2 / (2 * sigma**2))

y = model(theta)

plt.plot(x, y)
plt.xlabel('n * beam size')
plt.ylabel('beam profile')
plt.show()
