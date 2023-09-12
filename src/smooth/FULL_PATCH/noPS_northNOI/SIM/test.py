import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

m = np.load('./145.npy')
hp.mollview(m[0])
plt.show()
