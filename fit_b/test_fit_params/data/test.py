import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

# m = np.load('./map/pcfn_3_0.npy')[1]
m = np.load('./ps/ps3.npy')[1]
hp.gnomview(m)
plt.show()
