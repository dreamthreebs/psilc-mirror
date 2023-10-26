import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

mask = np.load('./north/BINMASKG.npy')

# smoothed_mask = hp.smoothing(mask, fwhm=np.deg2rad(5))

# hp.orthview(smoothed_mask, rot=[100,50,0], half_sky=True)
# hp.orthview(mask, rot=[100,50,0], half_sky=True)
# new_mask = np.zeros_like(mask)
# new_mask[smoothed_mask>0.05] = 1
# hp.orthview(new_mask, rot=[100,50,0], half_sky=True,title='next mask')
# hp.orthview(new_mask-mask, rot=[100,50,0], half_sky=True,title='residual')
# np.save('./north_smooth/new_bin_mask', new_mask)

mask1 = np.load('./north/APOMASKC1_10.npy')
new_mask = np.ones_like(mask)
new_mask[mask1<1] = 0


hp.orthview(mask, rot=[100,50,0], half_sky=True)
hp.orthview(mask1, rot=[100,50,0], half_sky=True)
hp.orthview(new_mask, rot=[100,50,0], half_sky=True,title='residual')
hp.orthview(mask-new_mask, rot=[100,50,0], half_sky=True,title='residual')
plt.show()

np.save('./north_smooth/BINMASKG.npy', new_mask)

m = np.load('./north_smooth/BINMASKG.npy')

hp.orthview(m, rot=[100,50,0], half_sky=True)
plt.show()
