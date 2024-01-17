import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

cov = np.load(f'./1.npy')
cov1 = np.load(f'../cov2/1.npy')
print(f'{cov=}')
print(f'{cov1=}')
