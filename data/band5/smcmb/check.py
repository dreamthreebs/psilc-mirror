import numpy as np
import healpy as hp
import matplotlib.pyplot as plt


m = np.load('./data.npy')
print(f'{m.shape=}')
hp.orthview(m[0], rot=[100,50,0], half_sky=True)
hp.orthview(m[4], rot=[100,50,0], half_sky=True)
plt.show()

diff_m = m[0]-m[4]
lmax=500
l = np.arange(lmax+1)
mask = np.load('../../../src/mask/north/APOMASKC1_15.npy')
fsky = np.sum(mask)/np.size(mask)
cl = hp.anafast(diff_m, lmax=lmax)
plt.semilogy(l*(l+1)*(cl)/(2*np.pi)/fsky)
plt.show()
