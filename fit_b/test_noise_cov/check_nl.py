import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

nstd = 1
nside = 512
npix = hp.nside2npix(nside)
lmax = 1000
l = np.arange(3*nside)

n_qu = np.random.normal(loc=0, scale=1, size=(3, npix))

nl_qu_1500 = hp.anafast(n_qu)

n_b = hp.alm2map(hp.map2alm(n_qu)[2], nside=nside)
nl_b_1500 = hp.anafast(n_b)



plt.loglog(l, nl_qu_1500[2, ])
plt.loglog(l, nl_b_1500)
plt.show()


