import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

# m = hp.read_map('./40_126.fits')
m = np.load('../psmask/psmask40_126.npy')
hp.mollview(m)
plt.show()
