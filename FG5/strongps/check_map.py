import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

m = np.load('./40.npy')[0]

hp.mollview(m)
plt.show()
