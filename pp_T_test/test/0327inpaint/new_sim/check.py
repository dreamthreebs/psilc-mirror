import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

m = np.load('./ps_sim_17.npy')

hp.mollview(m, xsize=3000)
plt.show()

