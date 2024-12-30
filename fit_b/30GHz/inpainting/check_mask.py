import numpy as np
import healpy as hp
import matplotlib.pyplot as plt


apo_ps_mask = np.load('./mask/apo_ps_mask.npy')
mask = hp.read_map('./mask/mask_add_edge.fits')

hp.orthview(mask, rot=[100,50,0], title='mask')
hp.orthview(apo_ps_mask, rot=[100,50,0], title='apo_ps_mask')
plt.show()
