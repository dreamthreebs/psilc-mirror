import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

nside=2048
m = np.load('./pcn.npy')
m_b = hp.alm2map(hp.map2alm(m)[2], nside=nside)
np.save('./pcn_b.npy', m_b)

