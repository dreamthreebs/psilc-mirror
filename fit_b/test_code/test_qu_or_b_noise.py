import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

nside = 1024
npix = hp.nside2npix(nside)

rlz_idx=0

m_qu = np.random.normal(loc=0, scale=1, size=(3,npix))
m_qu_b = hp.alm2map(hp.map2alm(m_qu)[2], nside=nside)
np.save(f'./qu_or_b_noise/m_qu_b_{rlz_idx}.npy', m_qu_b)

m_b = np.random.normal(loc=0, scale=1, size=(npix,))
np.save(f'./qu_or_b_noise/m_b_{rlz_idx}.npy', m_b)





