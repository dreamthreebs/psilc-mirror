import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

nside = 1024
lmax = 1024

m = np.zeros(hp.nside2npix(nside))
ipix = hp.ang2pix(nside=nside, theta=np.pi/2, phi=0)
m[ipix] = 1
m1 = hp.smoothing(m, fwhm=np.deg2rad(1), lmax=lmax)
vec = hp.ang2vec(theta=np.pi/2, phi=0 )

mask = np.zeros(hp.nside2npix(nside))
disc_ipix = hp.query_disc(nside=nside, vec=vec, radius=1.5*np.deg2rad(1))
mask[disc_ipix] = 1




hp.gnomview(m1*mask)
plt.show()


