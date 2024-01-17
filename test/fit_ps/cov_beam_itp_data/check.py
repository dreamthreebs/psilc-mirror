import numpy as np

cov = np.load('./lmax500rf0.6.npy')
print(f"{cov[:10,:10]=}")
