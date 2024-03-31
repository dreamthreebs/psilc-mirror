import numpy as np

cov = np.load('./cmb_cov_2048/r_1.5/270/0.npy')

# cov  = cov + 3.6 * np.eye(cov.shape[0])
print(f'{cov=}')
eigenvalues, eigenvectors = np.linalg.eigh(cov)
print(f'{eigenvalues=}')

eigenvalues[eigenvalues < 0] = 1e-6
reconstructed_cov = np.dot(eigenvectors * eigenvalues, eigenvectors.T)
eigenvalues, _ = np.linalg.eigh(reconstructed_cov)
print(f'{eigenvalues=}')
print(f'{reconstructed_cov=}')
print(f'{np.max(np.abs(reconstructed_cov-cov))=}')
np.save('./cmb_cov_2048/r_1.5/270/0.npy', reconstructed_cov)

# cov_tuple = np.linalg.svd(cov)
# print(f'{cov_tuple=}')
# inv_cov = np.linalg.solve(cov, np.eye(cov.shape[0]))
# # inv_cov = np.linalg.inv(cov)
# # inv_cov = np.linalg.pinv(cov)
# print(f'{inv_cov=}')
# I = cov @ inv_cov
# print(f'{I=}')
# is_correct = np.allclose(I, np.eye(cov.shape[0]), atol=1e-10, rtol=1e-9)

# print(is_correct)

