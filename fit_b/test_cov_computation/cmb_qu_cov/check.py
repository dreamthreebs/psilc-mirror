import numpy as np

cov_int = np.load('./0.npy')
cov_long = np.load('./0_long.npy')
has_nonzero = np.any(cov_long-cov_int)

print(f'{cov_int=}')
print(f'{cov_long=}')
print(f'{cov_long-cov_int=}')
print(f'{has_nonzero=}')
