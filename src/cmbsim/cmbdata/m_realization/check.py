import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

m = np.load('./0.npy')
lmax = 2000
l = np.arange(lmax+1)

cl = hp.anafast(m, lmax=lmax)

m1 = np.load('./1.npy')
cl1 = hp.anafast(m1, lmax=lmax)


plt.plot(l**2*cl[0])
plt.plot(l**2*cl1[0])
plt.show()
