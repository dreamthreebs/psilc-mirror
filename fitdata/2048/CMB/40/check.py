import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

lmax = 2000
l = np.arange(lmax+1)

m = np.load('./0.npy')
m1 = np.load('./55.npy')
print(f'{m.shape}')

# cl = hp.anafast(m, lmax=lmax)
# cl1 = hp.anafast(m1, lmax=lmax)

# plt.plot(l**2 * cl[0])
# plt.plot(l**2 * cl1[0])
# plt.show()

hp.mollview(m[0])
hp.mollview(m1[0])
plt.show()
