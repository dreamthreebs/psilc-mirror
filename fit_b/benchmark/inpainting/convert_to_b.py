import numpy as np
import healpy as hp

from pathlib import Path

rlz_idx=0
nside = 2048

q = hp.read_map(f'./output_1/q_{rlz_idx}.fits')
u = hp.read_map(f'./output_1/u_{rlz_idx}.fits')
t = np.zeros_like(q)

m_b = hp.alm2map(hp.map2alm([t,q,u])[2], nside=nside)

path_b = Path('./output_b_1')
path_b.mkdir(exist_ok=True, parents=True)
np.save(path_b / Path(f'{rlz_idx}.npy'), m_b)
