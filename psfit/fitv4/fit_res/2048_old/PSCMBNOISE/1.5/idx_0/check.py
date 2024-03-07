import numpy as np

a = np.load('./chi2dof.npy')
b = np.load('./fit_error.npy')
c = np.load('./norm_error.npy')
d = np.load('./norm_beam.npy')
e = np.load('../../../CMBNOISE/1.5/idx_0/norm_beam.npy')
print(f'{a}')
print(f'{b}')
print(f'{c}')
print(f'{d}')
print(f'{d-e}')

