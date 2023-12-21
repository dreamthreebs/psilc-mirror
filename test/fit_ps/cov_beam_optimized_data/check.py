import numpy as np

cov1 = np.load('./lmax350rf0.5.npy')
cov2 = np.load('./lmax350rf0.52.npy')

# np.set_printoptions(threshold=np.inf)
print(f'{np.max(cov1-cov2)}')
