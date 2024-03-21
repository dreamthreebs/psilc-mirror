import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

m = np.load('./100.npy')

nstd = np.load('../../../../FGSim/NSTDNORTH/2048/155.npy')
hp.mollview(nstd[1])
plt.show()

std = np.std(m[1])
print(f'{std=}')


