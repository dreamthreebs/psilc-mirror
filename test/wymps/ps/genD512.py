import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

ps = np.load('./d_2048_512.npy')
nstd = np.load('../../../FGSim/NSTDNORTH/40.npy')[0]

noise = nstd * np.random.normal(0,1,(hp.nside2npix(512)))

psnoise = ps + noise

hp.gnomview(psnoise)
plt.show()

np.save('./psnoise512.npy', psnoise)
