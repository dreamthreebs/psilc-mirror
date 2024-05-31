import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

lmax = 1999
l = np.arange(lmax+1)
nside = 2048

def dl_factorize(cl):
    return l*(l+1)*cl/(2*np.pi)

cn = np.load('../../fitdata/synthesis_data/2048/CMBNOISE/30/1.npy')
c = np.load('../../fitdata/2048/CMB/30/1.npy')
n = np.load('../../fitdata/2048/NOISE/30/1.npy')

cn_B = hp.alm2map(hp.map2alm(cn, lmax=lmax)[2], nside=nside)
c_B = hp.alm2map(hp.map2alm(c, lmax=lmax)[2], nside=nside)
n_B = hp.alm2map(hp.map2alm(n, lmax=lmax)[2], nside=nside)

cn_B_cl = hp.anafast(cn_B, lmax=lmax)
c_B_cl = hp.anafast(c_B, lmax=lmax)
n_B_cl = hp.anafast(n_B, lmax=lmax)

plt.plot(dl_factorize(cn_B_cl), label='cn_B')
plt.plot(dl_factorize(c_B_cl), label='c_B')
plt.plot(dl_factorize(n_B_cl), label='n_B')
plt.plot(dl_factorize(cn_B_cl - n_B_cl), label='debias noise')

plt.semilogy()
plt.legend()
plt.show()

