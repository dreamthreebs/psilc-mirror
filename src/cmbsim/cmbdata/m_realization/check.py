import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

m = np.load('./0.npy')
lmax = 2000
l = np.arange(lmax+1)

cl = hp.anafast(m, lmax=lmax)

m1 = np.load('./51.npy')
cl1 = hp.anafast(m1, lmax=lmax)

# hp.mollview(m[0])
# hp.mollview(m1[0])
# plt.show()

plt.plot(l**2*cl[2])
plt.plot(l**2*cl1[2])
plt.show()
