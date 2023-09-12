import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

cmb = np.load('./145.npy')
print(f'{cmb.shape = }')
for i in range(3):
    hp.mollview(cmb[i]);plt.show()
