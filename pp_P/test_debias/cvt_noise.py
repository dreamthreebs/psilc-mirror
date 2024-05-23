import numpy as np
import healpy as hp

from pathlib import Path

lmax = 400
nside = 256

path_E = Path('./noise/E')
path_B = Path('./noise/B')
path_E.mkdir(exist_ok=True, parents=True)
path_B.mkdir(exist_ok=True, parents=True)

for rlz_idx in range(1000):
    print(f'{rlz_idx=}')
    IQU = np.load(f'./noise/{rlz_idx}.npy')
    E = hp.alm2map(hp.map2alm(IQU, lmax=lmax)[1], nside=nside)
    B = hp.alm2map(hp.map2alm(IQU, lmax=lmax)[2], nside=nside)
    np.save(path_E / Path(f'{rlz_idx}.npy'), E)
    np.save(path_B / Path(f'{rlz_idx}.npy'), B)
