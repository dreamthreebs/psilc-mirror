import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

lon = 0
lat = 0


ps = np.load('./ps/ps_map_4.0.npy')
hp.gnomview(ps)
plt.show()
