import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

noise = np.load('./0.npy')[0]
nstd = np.load('../../../../FGSim/NSTDNORTH/40.npy')[0]
print(f'{nstd[0]=}')
std = np.std(noise)
print(f'{std}')
