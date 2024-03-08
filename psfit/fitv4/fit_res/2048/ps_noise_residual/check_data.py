import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

m = np.load('./0.npy')
print(f'{m.shape}')
mask = np.load('../../../../../src/mask/north/BINMASKG2048.npy')

hp.orthview(m*mask, rot=[100,50,0], half_sky=True, min=-1, max=1)
plt.show()
