import numpy as np

data = np.load('./data.npy')
print(f'{data.shape=}')
cov = np.cov(data, rowvar=False)
print(f'{cov=}')
print(f'{cov.shape=}')
np.save('./cov/1.npy', cov)
