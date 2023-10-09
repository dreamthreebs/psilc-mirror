import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

mask = np.load("./apo_circle_mask2048C1_8.npy")
fsky = np.sum(mask) / np.size(mask)
m = hp.smoothing(mask, fwhm=np.deg2rad(1)/60)
# print(f'{fsky=}')
hp.mollview(m)
plt.show()

