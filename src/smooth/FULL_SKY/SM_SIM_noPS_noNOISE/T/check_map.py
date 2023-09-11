import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

m = np.load('./cmbfg.npy')
print(m.shape)
# hp.mollview(m[0], norm='hist');plt.show()
