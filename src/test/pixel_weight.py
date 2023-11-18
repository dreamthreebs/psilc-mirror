import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

nside = 512
npix = hp.nside2npix(nside)
m = np.arange(npix)
lmax=100

alm = hp.map2alm(m, lmax, use_pixel_weights=True)

