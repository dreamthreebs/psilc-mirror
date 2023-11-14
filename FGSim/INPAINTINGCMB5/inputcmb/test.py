import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

# m = hp.read_map('./40.fits', field=(0,1,2))
# print(f'{m.shape=}')

lmax = 1000
bl1 = hp.gauss_beam(fwhm=np.deg2rad(1), lmax=lmax)
bl2 = hp.gauss_beam(fwhm=np.deg2rad(1), lmax=lmax, pol=True)[:,0]

bl_diff = bl1-bl2

