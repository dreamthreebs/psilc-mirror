import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from pathlib import Path

nside = 2048
npix = hp.nside2npix(nside)

nstd = 0.1
ipix_idx = np.load('./data/pix_idx.npy')

rlz_idx=0

noise = nstd * np.random.normal(loc=0, scale=1, size=(3,npix))
m_b = hp.alm2map(hp.map2alm(noise)[2], nside=nside)
path_noise = Path('noise_sim')
path_noise.mkdir(exist_ok=True, parents=True)

np.save(path_noise / Path(f'{rlz_idx}.npy'), m_b[ipix_idx])
