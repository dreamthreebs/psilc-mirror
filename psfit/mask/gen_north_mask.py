import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

beam = 63
lmax = 1000
m = np.load('../../FGSim/PSNOISE/1024/40psnoise.npy')[0]
mask = np.load('../../src/mask/north/BINMASKG1024.npy')
# hp.mollview(mask)
# hp.orthview(mask, rot=[100,50,0], half_sky=True, title='origin binary mask')

sm_mask = hp.smoothing(mask, fwhm=2 * np.deg2rad(beam)/60, lmax=lmax)
new_mask = np.zeros_like(mask)
new_mask[sm_mask > 0.01] = 1
# hp.orthview(new_mask, rot=[100,50,0], half_sky=True, title='smoothed new binary mask')
# hp.orthview(sm_mask, rot=[100,50,0], half_sky=True, title='smooth mask')
# hp.orthview(new_mask-mask, rot=[100,50,0], half_sky=True, title='mask residual')
hp.orthview(new_mask * m, rot=[100,50,0], half_sky=True, title='point source', min=-30, max=30)
hp.orthview(mask * m, rot=[100,50,0], half_sky=True, title='point source', min=-30, max=30)

plt.show()


