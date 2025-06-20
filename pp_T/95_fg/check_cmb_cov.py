import numpy as np

cov_1k = np.load('./cmb_cov_1k/1.npy')
cov_2k = np.load('./cmb_cov_1k6/1.npy')
# cov_2d5k = np.load('./cmb_cov_2d5k/1.npy')
# cov_3k = np.load('./cmb_cov_3k/1.npy')

print(f'{cov_1k=}')
print(f'{cov_2k=}')
# print(f'{cov_2d5k=}')
# print(f'{cov_3k=}')
