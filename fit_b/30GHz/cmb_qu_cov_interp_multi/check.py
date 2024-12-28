import numpy as np

cov_int = np.load('../cmb_qu_cov_interp/14_6_8.npy')
cov_long = np.load('./14_6_8.npy')
has_nonzero = np.any(cov_long-cov_int)

print(f'{cov_int=}')
print(f'{cov_long=}')
print(f'{cov_long-cov_int=}')
print(f'{has_nonzero=}')
