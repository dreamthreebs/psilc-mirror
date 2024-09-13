import numpy as np

pcn_list = []
pcfn_list = []
for rlz_idx in range(100):
    pcn_err = np.load(f'./pcn_params/fit_qu/idx_1/P_{rlz_idx}.npy')
    pcfn_err = np.load(f'./pcfn_params/fit_qu/idx_1/fit_err_P_{rlz_idx}.npy')
    pcn_list.append(pcn_err)
    pcfn_list.append(pcfn_err)
print(f'{pcn_list=}')
print(f'{pcfn_list=}')
