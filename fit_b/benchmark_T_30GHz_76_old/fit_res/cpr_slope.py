import numpy as np

for rlz_idx in range(200):
    pcfn_slope = np.load(f'./slope_pcfn/{rlz_idx}.npy')
    cmb_slope = np.load(f'./pcfn_fit_qu/eblc_slope_cmb/{rlz_idx}.npy')
    print(f'{pcfn_slope=}, {cmb_slope=}')



