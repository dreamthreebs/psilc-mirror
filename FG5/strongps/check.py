import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

m = np.load('./40.npy')

hp.mollview(m[0], norm='hist')
plt.show()
