import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

# m = np.load('../ps_noise_residual/0.npy')
m = np.load('../../../../../fitdata/synthesis_data/2048/PSNOISE/40/0.npy')[0]
mask = np.load('../../../../../src/mask/north/BINMASKG2048.npy')
mask_no_edge = np.load('./no_edge_mask/C1_2.npy')

hp.orthview(m*mask, rot=[100,50,0], title='origin', half_sky=True)
hp.orthview(m*mask_no_edge, rot=[100,50,0], title='no edge', half_sky=True)
plt.show()
