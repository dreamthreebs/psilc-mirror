import numpy as np

vecs = np.load('./vecs.npy')
print(f'{vecs=}')

contains_nan = np.any(np.isnan(vecs))
print(contains_nan)
