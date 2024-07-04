import numpy as np
import matplotlib.pyplot as plt

x = np.array([1,2,3,4])
y = np.array([5,9,6,4])

# Generate a random 4x4 covariance matrix
np.random.seed(42)  # Seed for reproducibility
random_matrix = np.random.randn(4, 4)
random_cov_matrix = np.dot(random_matrix, random_matrix.T)
std_devs = np.sqrt(np.diag(random_cov_matrix)[:2])


plt.errorbar(x, y, xerr=std_devs[0], yerr=std_devs[1], fmt='o', label='Data with Error Bars')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.title('Data Points with Error Bars')
plt.legend()
plt.show()

