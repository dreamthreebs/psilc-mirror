import numpy as np

cov = np.load('./1.npy')
cov1 = np.load('../../cmb_cov_beam_256/cov/1.npy')
print(f'{cov=}')
print(f'{cov1=}')
