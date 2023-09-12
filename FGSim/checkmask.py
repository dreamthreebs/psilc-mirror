import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

north = np.load('./northMask.npy')
south = np.load('./southMask.npy')
hp.mollview(south)
plt.show()
