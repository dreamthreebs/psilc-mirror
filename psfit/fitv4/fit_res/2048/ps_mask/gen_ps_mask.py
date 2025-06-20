import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

m = np.load('./new_mask/apo_C1_3_apo_3.npy')

hp.orthview(m, rot=[100,50,0], half_sky=True)
plt.show()

m[m < 1] = 0

hp.orthview(m, rot=[100,50,0], half_sky=True)
plt.show()

np.save('./new_mask/BIN_C1_3_C1_3.npy', m)

