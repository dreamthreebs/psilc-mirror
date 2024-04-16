import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pymaster as nmt

rlz_idx = 0
mask_raw = hp.read_map(f'./{rlz_idx}.fits', field=0)
mask_no_edge = np.load('../../../../../../../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5.npy')
mask_no_edge_apo = np.load('../../../../../../../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5APO_5.npy')

# hp.orthview(mask_raw, rot=[100,50,0], half_sky=True)
# hp.orthview(mask_no_edge, rot=[100,50,0], half_sky=True)

aposcale = 1

apo_ps = nmt.mask_apodization(mask_in=mask_raw, aposize=aposcale, apotype='C1')

# hp.orthview(apo_ps, rot=[100,50,0], half_sky=True)
# hp.orthview(apo_ps*mask_no_edge, rot=[100,50,0], half_sky=True)
# hp.orthview(apo_ps*mask_no_edge_apo, rot=[100,50,0], half_sky=True)
# plt.show()

np.save(f'./apodize_mask/{rlz_idx}_2.npy', apo_ps)

