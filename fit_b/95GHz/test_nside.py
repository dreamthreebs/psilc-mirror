import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

# Define the resolution parameter nside
nside = 200
nside_out = 187

# Create a random map with nside = 200
# map_data = np.random.randn(hp.nside2npix(nside))
map_data = np.random.randn(12*nside**2)
map_data_2 = hp.alm2map(hp.map2alm(map_data, lmax=300), nside=nside_out)

# Plot the map in Mollweide projection
hp.mollview(map_data, title="Random Healpy Map (nside=200)", cmap='viridis')
hp.mollview(map_data_2, title="Random Healpy Map (nside=200)", cmap='viridis')
plt.show()

