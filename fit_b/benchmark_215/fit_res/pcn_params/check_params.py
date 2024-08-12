import numpy as np

rlz_idx = 1
fit_A = np.load(f'./fit_b/idx_0/fit_P_{rlz_idx}.npy')
fit_A_1 = np.load(f'./fit_b_1/idx_0/fit_P_{rlz_idx}.npy')
fit_A_QU = np.load(f'./fit_qu/idx_0/fit_P_{rlz_idx}.npy')

print(f'{fit_A=}, {fit_A_1=}, {fit_A_QU=}')
