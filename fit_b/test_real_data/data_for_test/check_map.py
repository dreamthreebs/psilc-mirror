import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

m1 = np.load('./map_from_gen.npy')
m2 = np.load(f'./map_from_slurm.npy')
fuck = np.max(m1-m2)
print(f'{fuck=}')

# hp.mollview(m1)
# hp.mollview(m2)
# hp.mollview(m1-m2)
# plt.show()
