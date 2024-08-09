import numpy as np

flux_idx = 10

# P_true = np.load(f'./32core/idx_{flux_idx}/P_0.npy')
# phi_true = np.load(f'./32core/idx_{flux_idx}/phi_0.npy')

P_err_list = []
phi_err_list = []

for rlz_idx in range(1, 90):

    # if rlz_idx==76:
        # print(f'{rlz_idx=}')
        # print(f'{P_true=}, {phi_true=}')
    P = np.load(f'./idx_{flux_idx}/fit_P_{rlz_idx}.npy')
    P_err = np.load(f'./idx_{flux_idx}/fit_err_P_{rlz_idx}.npy')
    P_err_list.append(P_err)
    # if P_err < 20:
        # print(f'{rlz_idx=}')
    phi = np.load(f'./idx_{flux_idx}/fit_phi_{rlz_idx}.npy')
    phi_err = np.load(f'./idx_{flux_idx}/fit_err_phi_{rlz_idx}.npy')
    phi_err_list.append(phi_err)
    print(f'{P=},{P_err=},{phi=},{phi_err=}')

P_err_mean = np.mean(P_err_list)
P_err_std = np.std(P_err_list)
phi_err_mean = np.mean(phi_err_list)
phi_err_std = np.std(phi_err_list)

print(f'{P_err_mean=}, {P_err_std=}, {phi_err_mean=}, {phi_err_std=}')


