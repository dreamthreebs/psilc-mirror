import numpy as np

cov1 = np.load('./1.npy')
cov2 = np.load('../cov_itp_256/1.npy')

print(cov1-cov2)
print(np.max(np.abs(cov1-cov2)))

