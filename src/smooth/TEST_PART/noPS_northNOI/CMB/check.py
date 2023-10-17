import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

m = np.load('./145.npy')
hp.mollview(m[0])
plt.show()
