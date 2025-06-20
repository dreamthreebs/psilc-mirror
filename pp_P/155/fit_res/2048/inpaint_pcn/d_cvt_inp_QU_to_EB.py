import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from eblc_base import EBLeakageCorrection

lmax = 1999
nside = 2048
threshold = 3

rlz_idx = 0

ori_mask = np.load('../../../../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5.npy')

Q = hp.read_map(f'./{threshold}sigma/QU/Q_output/{rlz_idx}.fits')
U = hp.read_map(f'./{threshold}sigma/QU/U_output/{rlz_idx}.fits')
I = np.zeros_like(Q)

# hp.orthview(Q*ori_mask, rot=[100,50,0], half_sky=True)
# plt.show()

obj_eblc = EBLeakageCorrection(m=np.array([I,Q,U]), lmax=lmax, nside=nside, mask=ori_mask, post_mask=ori_mask)
_,_,cln_b = obj_eblc.run_eblc()

crt_e = hp.alm2map(hp.map2alm(np.array([I,Q,U]), lmax=lmax)[1], nside=nside) * ori_mask

hp.write_map(f'./{threshold}sigma/QU/B/{rlz_idx}.fits', cln_b, overwrite=True)
hp.write_map(f'./{threshold}sigma/QU/E/{rlz_idx}.fits', crt_e, overwrite=True)

# hp.orthview(cln_b, rot=[100,50,0], title='clean b')
# hp.orthview(crt_e, rot=[100,50,0], title='currupted e')
# plt.show()






