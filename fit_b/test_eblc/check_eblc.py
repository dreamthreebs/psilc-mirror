import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from eblc_base import EBLeakageCorrection

nside = 2048
mask = np.load('../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5.npy')
m = np.load('../../fitdata/synthesis_data/2048/CMBNOISE/270/1.npy')


lmax = 2000
obj = EBLeakageCorrection(m=m, lmax=lmax, nside=nside, mask=mask, post_mask=mask)
_,_, cln_b_2k = obj.run_eblc()

lmax = 3*nside-1
obj = EBLeakageCorrection(m=m, lmax=lmax, nside=nside, mask=mask, post_mask=mask)
_,_, cln_b_6k = obj.run_eblc()

hp.orthview(cln_b_2k, rot=[100,50,0], title='eblc 2k')
hp.orthview(cln_b_6k, rot=[100,50,0], title='eblc 6k')
hp.orthview(cln_b_6k-cln_b_2k, rot=[100,50,0], title='res')
plt.show()
