import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

lmax = 1500
nside = 512

c = np.load('./cmb/0.npy')
cfn = np.load('./cfn/0.npy')
cn = np.load('./cn/0.npy')

c_b  = hp.alm2map(hp.map2alm(c, lmax=lmax)[2], nside=nside)
cfn_b = hp.alm2map(hp.map2alm(cfn, lmax=lmax)[2], nside=nside)

hp.mollview(c_b, title='c_b')
hp.mollview(cfn_b, title='cfn_b')
plt.show()




