import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

m = np.load('./cn/1.npy')
hp.mollview(m[1])
plt.show()

