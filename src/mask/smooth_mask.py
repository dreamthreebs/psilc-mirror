import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

mask = np.load('./north/BINMASKG.npy')

smoothed_mask = hp.smoothing(mask, fwhm=np.deg2rad(5))

hp.orthview(smoothed_mask, rot=[100,50,0], half_sky=True)
hp.orthview(mask, rot=[100,50,0], half_sky=True)
mask[smoothed_mask>0.05] = 1
hp.orthview(mask, rot=[100,50,0], half_sky=True,title='next mask')
plt.show()
