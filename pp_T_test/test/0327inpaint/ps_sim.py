import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

# Step 1: Set up the map parameters
nside = 2048
npix = hp.nside2npix(nside) # Number of pixels in the map

# Step 2: Generate random positions for point sources
n_sources = 1000 # Number of point sources you want to generate
source_indices = np.random.randint(0, npix, n_sources) # Random pixel indices for point sources
np.save('source_indices.npy', source_indices)

# Step 3: Assign flux densities to these point sources using a Poisson distribution
lambda_param = 100000 # Mean of the Poisson distribution
flux_densities = np.random.poisson(lambda_param, n_sources)
np.save('flux_densities.npy', flux_densities)

# Step 4: Create the map and populate it with the point sources
sky_map = np.zeros(npix)
np.add.at(sky_map, source_indices, flux_densities) # Add flux densities to the corresponding pixels

sm_map = hp.smoothing(sky_map, fwhm=np.deg2rad(17)/60, lmax=2000)
# Plotting the map
hp.mollview(sky_map, title='Point Source Map with Poisson-distributed Flux Densities', unit='Flux Density', norm='hist', cmap='inferno')
hp.mollview(sm_map, title='Point Source smooth map', unit='muKCMB', norm='hist', cmap='inferno')
np.save('./ps_sim.npy', sm_map)
plt.show()

