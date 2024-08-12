import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from pathlib import Path
from eblc_base import EBLeakageCorrection

rlz_idx=0
nside = 2048
mask = hp.read_map(f'./mask/mask_1.fits')
m_q = hp.read_map(f'./input/q_{rlz_idx}.fits')
m_u = hp.read_map(f'./input/u_{rlz_idx}.fits')
m_t = np.zeros_like(m_q)

obj = EBLeakageCorrection(np.array([m_t, m_q, m_u]), lmax=3*nside-1, nside=nside, mask=mask, post_mask=mask)
_,_,cln_b = obj.run_eblc()

path_b = Path('./input_b')
path_b.mkdir(exist_ok=True, parents=True)
hp.write_map(path_b / Path(f'{rlz_idx}.fits'), cln_b, overwrite=True)



