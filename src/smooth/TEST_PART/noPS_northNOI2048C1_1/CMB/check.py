import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from ftbutils.plot_healpy_maps import plot_scalar_map_mollview

m = np.load('./145.npy')
plot_scalar_map_mollview(m[0])
