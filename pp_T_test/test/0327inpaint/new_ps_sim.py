import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

# Step 1: Set up the map parameters
lmax = 2000
nside = 2048
npix = hp.nside2npix(nside) # Number of pixels in the map
# m = np.zeros(npix)

ipix_list = []
for i in np.linspace(-178.5, 178.5, 100):
    print(f'{i=}')
    ipix = hp.ang2pix(nside=nside, theta=i, phi=0, lonlat=True)
    print(f'{ipix=}')
    ipix_list.append(ipix)

    # hp.mollview(m)
    # hp.projscatter(theta=i, phi=0, lonlat=True)
    # plt.show()

ipix_arr = np.asarray(ipix_list)
print(f'{ipix_arr=}')
n_sources = len(ipix_arr)
source_indices = ipix_arr
np.save('./new_sim/source_indices.npy', source_indices)

lambda_param = 100000 # Mean of the Poisson distribution
flux_densities = np.random.poisson(lambda_param, n_sources)
np.save('./new_sim/flux_densities.npy', flux_densities)

# Step 4: Create the map and populate it with the point sources
sky_map = np.zeros(npix)
np.add.at(sky_map, source_indices, flux_densities) # Add flux densities to the corresponding pixels

sm_map = hp.smoothing(sky_map, fwhm=np.deg2rad(17)/60, lmax=2000)
# Plotting the map
hp.mollview(sky_map, title='Point Source Map with Poisson-distributed Flux Densities', unit='Flux Density', norm='hist', cmap='inferno')
hp.mollview(sm_map, title='Point Source smooth map', unit='muKCMB', norm='hist', cmap='inferno')
np.save('./new_sim/ps_sim_17.npy', sm_map)
plt.show()

