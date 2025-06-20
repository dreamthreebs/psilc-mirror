import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

nside = 128
npix  = hp.nside2npix(nside)
m = np.zeros((3, npix))
idx = hp.ang2pix(nside, np.deg2rad(45), np.deg2rad(45))

m[:, idx] = np.random.uniform(-10, 10, 3)

# smoothing by healpy

healpy_m = hp.smoothing(m, np.deg2rad(5))
hp.gnomview(healpy_m[1], rot=(45, 45), sub=121)
hp.gnomview(healpy_m[2], rot=(45, 45), sub=122)

# smoothing by lugwid

P = m[1, idx] + 1j * m[2, idx]
_, iphi = hp.pix2ang(nside, idx)
P = P * np.exp(2j * iphi)

ivec = np.array(hp.pix2vec(nside, idx))
vec = np.vstack(hp.pix2vec(nside, np.arange(npix)))

cost = ivec @ vec
half_t2 = 1 - cost
sigma = np.deg2rad(5) / 2.355
profile = np.exp(-half_t2 / sigma**2)

inv_norm_factor = np.sum(profile)
profile /= inv_norm_factor

lugwidP = P * profile
_, phi = hp.pix2ang(nside, np.arange(npix))
QU = lugwidP * np.exp(-2j * phi)
Q = QU.real
U = QU.imag

fig = plt.figure()
hp.gnomview(Q, rot=(45, 45), sub=121, fig=fig)
hp.gnomview(U, rot=(45, 45), sub=122, fig=fig)
plt.show()




