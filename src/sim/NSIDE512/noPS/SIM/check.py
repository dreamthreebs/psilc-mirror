import numpy as np
import healpy as hp
import matplotlib.pyplot as plt


m = np.load('./270.npy')
hp.mollview(m[1])
plt.show()
