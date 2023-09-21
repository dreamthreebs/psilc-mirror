import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

north = np.load('./northMask.npy')
south = np.load('./southMask.npy')
fg = np.load('./FG/145.npy')
hp.mollview(fg[0])
plt.show()
