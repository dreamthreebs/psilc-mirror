import numpy as np

cov = np.load('./1.npy')
print(f'{cov[-10:,-10:]=}')
