import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from pathlib import Path
from eblc_base_slope import EBLeakageCorrection

rlz_idx=0
nside = 2048

q = np.load(f'./3sigma/map_q_{rlz_idx}.npy')
u = np.load(f'./3sigma/map_u_{rlz_idx}.npy')
t = np.zeros_like(q)
cln_b = hp.alm2map(hp.map2alm([t,q,u])[2], nside=nside)

path_b = Path('./3sigma/B')
path_b.mkdir(exist_ok=True, parents=True)
np.save(path_b / Path(f'{rlz_idx}.npy'), cln_b)



