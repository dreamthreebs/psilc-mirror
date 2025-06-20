import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt

from eblc_base import EBLeakageCorrection

threshold = 3
rlz_idx = 0
lmax = 1999
nside = 2048

q = hp.read_map(f'./{threshold}sigma/input/Q/{rlz_idx}.fits')
u = hp.read_map(f'./{threshold}sigma/input/U/{rlz_idx}.fits')
i = np.zeros_like(q)

orimask = np.load('../../../../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5.npy')
mask = hp.read_map(f'./{threshold}sigma/bin_mask/{rlz_idx}.fits')

masked_q = q * mask
masked_u = u * mask

# hp.orthview(q*mask, rot=[100,50,0], title='q', half_sky=True, xsize=1600)
# hp.orthview(q*orimask, rot=[100,50,0], title='q orimask', half_sky=True, xsize=1600)
# hp.orthview(q*orimask - q*mask, rot=[100,50,0], title='q orimask - mask', half_sky=True, xsize=1600)
# hp.orthview(u*mask, rot=[100,50,0], title='u', half_sky=True, xsize=1600)
# hp.orthview(mask, rot=[100,50,0], title='mask', half_sky=True)
# plt.show()

alm_i, alm_e, alm_b = hp.map2alm([i, masked_q, masked_u], lmax=lmax)
crt_e = hp.alm2map(alm_e, nside=2048) * mask

obj_eblc = EBLeakageCorrection(m=[i, masked_q, masked_u] , lmax=lmax, nside=nside, mask=mask, post_mask=mask)
_,_,cln_b = obj_eblc.run_eblc()

hp.write_map(f'./{threshold}sigma/EB/E_input/{rlz_idx}.fits', crt_e, overwrite=True)
hp.write_map(f'./{threshold}sigma/EB/B_input/{rlz_idx}.fits', cln_b, overwrite=True)

# hp.orthview(crt_e, rot=[100,50,0], title='E', half_sky=True)
# hp.orthview(cln_b, rot=[100,50,0], title='B', half_sky=True)
# plt.show()

