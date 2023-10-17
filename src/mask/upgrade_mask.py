import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

mask = np.load('./north/BINMASKG.npy')

ugmask = hp.ud_grade(mask, nside_out=2048)
print(f'{ugmask.shape}')

hp.orthview(ugmask, rot=[100,50,0], half_sky=True)
plt.show()

np.save('./north/BINMASKG2048.npy', ugmask)
