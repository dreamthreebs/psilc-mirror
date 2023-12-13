import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

ps = np.load('../ps_maps/lon0lat0.npy')
nstd = np.load('../../../../FGSim/NSTDNORTH/2048/40.npy')[0]

nside = 2048
n_pix = hp.nside2npix(nside)
noise = nstd * np.random.normal(loc=0, scale=1, size=(n_pix))

m = ps + noise
np.save('ps_ns.npy', m)
hp.gnomview(m)
plt.show()
