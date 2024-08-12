import numpy as np
import healpy as hp

nside = 2048
m_ps = np.load('../data/ps/ps.npy')

m_b = hp.alm2map(hp.map2alm(m_ps)[2], nside=nside)

np.save('../data/ps/ps_b.npy', m_b)

