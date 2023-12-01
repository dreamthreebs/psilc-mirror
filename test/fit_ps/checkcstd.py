import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

m = np.load('../../inpaintingdata/CMB8/40.npy')[0]
std = np.std(m, ddof=1)
print(f'{std=}')
