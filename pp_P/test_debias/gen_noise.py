import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from pathlib import Path

lmax = 400
nside = 256
beam = 63

nstd = np.load('../../FGSim/NSTDNORTH/256/40.npy')
path_noise = Path(f'./noise')
path_noise.mkdir(exist_ok=True, parents=True)

for rlz_idx in range(1000):
    print(f'{rlz_idx=}')

    noise = nstd * np.random.normal(loc=0, scale=1, size=(nstd.shape[0], nstd.shape[1]))
    np.save(f'./noise/{rlz_idx}.npy', noise)

