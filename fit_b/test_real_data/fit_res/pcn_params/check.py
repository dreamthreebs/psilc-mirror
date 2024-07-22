import numpy as np

for rlz_idx in range(100):

    P = np.load(f'./idx_0/fit_P_{rlz_idx}.npy')
    P_err = np.load(f'./idx_0/fit_P_err_{rlz_idx}.npy')
    phi = np.load(f'./idx_0/fit_phi_{rlz_idx}.npy')
    phi_err = np.load(f'./idx_0/fit_phi_err_{rlz_idx}.npy')
    print(f'{P=}, {P_err=}, {phi=}, {phi_err=}')
