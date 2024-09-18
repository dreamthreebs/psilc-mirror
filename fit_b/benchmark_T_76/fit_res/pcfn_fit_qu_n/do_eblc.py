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
mask = np.load('../../../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5.npy')
slope_in = np.load(f'../pcfn_fit_qu/eblc_slope/{rlz_idx}.npy')

obj = EBLeakageCorrection(m=np.asarray([t,q,u]), lmax=3*nside-1, nside=nside, mask=mask, post_mask=mask, slope_in=slope_in)
_,_,cln_b = obj.run_eblc()


path_b = Path('./3sigma/B_eblc')
path_b.mkdir(exist_ok=True, parents=True)
np.save(path_b / Path(f'{rlz_idx}.npy'), cln_b)



