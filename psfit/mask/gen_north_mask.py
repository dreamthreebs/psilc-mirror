import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

mask = np.load('../../src/mask/north/BINMASKG2048.npy')
hp.mollview(mask)
# hp.orthview(mask, rot=[100,50,0], half_sky=True)
plt.show()
