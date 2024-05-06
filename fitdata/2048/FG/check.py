import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

m = np.load('./155/fg.npy')
hp.mollview(m[1],title='Q', min=-30, max=90)
hp.mollview(m[2],title='U', min=-44, max=53)
plt.show()
