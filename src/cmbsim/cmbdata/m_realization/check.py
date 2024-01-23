import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

m = np.load('./0.npy')
lmax = 2000
l = np.arange(lmax+1)

cl = hp.anafast(m, lmax=lmax)
plt.plot(l**2*cl[0])
plt.show()
