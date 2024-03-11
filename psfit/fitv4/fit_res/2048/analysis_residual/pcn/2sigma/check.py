import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

m = np.load('./ps_res_avg.npy')
print(f'{m.shape=}')

hp.orthview(m, rot=[100,50,0], half_sky=True, min=-1, max=1)
plt.show()

