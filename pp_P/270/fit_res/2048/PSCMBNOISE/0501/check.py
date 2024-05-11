import numpy as np

def check_err():
    q_amp_err = np.load('./idx_1/q_amp_err.npy')
    u_amp_err = np.load('./idx_1/u_amp_err.npy')
    print(f'{q_amp_err=}')
    print(f'{u_amp_err=}')


def check_chi2dof():
    x = np.load('./idx_64/chi2dof.npy')[1:100]
    mean_x = np.mean(x)
    print(f'{x=}')
    print(f'{mean_x=}')

check_chi2dof()
