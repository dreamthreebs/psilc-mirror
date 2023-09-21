import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

# Initialize parameters
nside = 512

# Define center of disc in Galactic coordinates (l, b) = (0, 90)
vec = hp.ang2vec(np.radians(90), np.radians(0))

# Radius of circle in radians (47 degrees)
radius = np.radians(47)

# Use query_disc to generate indices for pixels within the circle
ind = hp.query_disc(nside, vec, radius)

# Create mask map
mask_map = np.zeros(hp.nside2npix(nside), dtype=bool)
mask_map[ind] = True

# Visualize
fsky=np.sum(mask_map)/np.size(mask_map)
print(f'{fsky}')
hp.mollview(mask_map, title="47-degree Circle Mask")

plt.show()


