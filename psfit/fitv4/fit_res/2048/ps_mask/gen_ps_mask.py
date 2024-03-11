import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

m = np.load('./north/APOMASKC_8.npy')

# hp.orthview(m, rot=[100,50,0], half_sky=True)
# plt.show()

m[m < 1] = 0

# hp.orthview(m, rot=[100,50,0], half_sky=True)
# plt.show()

np.save('./no_edge_mask/C1_8.npy', m)

