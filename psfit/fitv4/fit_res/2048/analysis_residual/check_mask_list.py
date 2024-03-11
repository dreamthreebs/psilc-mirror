import numpy as np

for rlz_idx in range(100):
    mask_list = np.load(f'../ps_cmb_noise_residual/2sigma/mask{rlz_idx}.npy')
    print(f'{rlz_idx=}, {mask_list=}')
