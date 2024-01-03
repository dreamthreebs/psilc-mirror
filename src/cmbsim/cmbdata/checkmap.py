import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

m = np.load('./cmbmap.npy')
hp.mollview(m[1])

plt.show()
