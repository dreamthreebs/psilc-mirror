import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

nstd = 0.23
nside = 2048
lmax = 2000
npix = hp.nside2npix(nside)

noise = nstd * np.random.normal(loc=0, scale=1, size=(3,npix))

noise_b_2k = hp.alm2map(hp.map2alm(noise, lmax=lmax)[2], nside=nside)
noise_b_6k = hp.alm2map(hp.map2alm(noise)[2], nside=nside)

hp.gnomview(noise_b_2k, title='lmax=2k')
hp.gnomview(noise_b_6k, title='lmax=3*nside-1')
plt.show()

