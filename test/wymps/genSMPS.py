import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

nside = 512
lmax = 2000
npix = hp.nside2npix(nside)
beam = 63
sigma = np.deg2rad(beam) / 60 / np.sqrt(8*np.log(2))

nstd = np.load('../../FGSim/NSTDNORTH/40.npy')[0]
noise = nstd * np.random.normal(0,1,(npix))

m = np.zeros(npix)

ipix = hp.ang2pix(nside=nside, theta=0, phi=0, lonlat=True)
vec = hp.pix2vec(nside=nside, ipix=ipix)
print(f'{ipix=}')
norm = 1e6
m[ipix] = norm

# hp.gnomview(m)
# plt.show()

sm_m = hp.smoothing(m, fwhm=np.deg2rad(beam)/60, lmax=lmax)

m = sm_m

hp.gnomview(m)
plt.show()

np.save('./ps/512.npy', m)



