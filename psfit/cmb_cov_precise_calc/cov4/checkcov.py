import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

# cov = np.load('../../cmb_cov_calc/cov/1.npy')
cov = np.load('../cov4/1.npy')

# print(f'{cov[:10,:10]=}')
print(f'{cov[-30:,-30:]=}')
diag = np.diag(cov)
print(f'{np.min(diag)=}')
print(f'{np.mean(diag)=}')
