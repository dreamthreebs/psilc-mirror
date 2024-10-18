import numpy as np

cov_1 = np.load(f'./cmb_cov_pol/1.npy')
# cov_2 = np.load(f'../30/cmb_cov_2048/r_1.5/1.npy')
cov_2 = np.load(f'./cmb_fg_cov/1.npy')

print(f'{cov_1=}')
print(f'{cov_2=}')
print(f'{cov_1-cov_2=}')
