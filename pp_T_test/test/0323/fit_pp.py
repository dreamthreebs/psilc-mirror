import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from iminuit import Minuit


nside = 2048
lmax = 2000
npix = hp.nside2npix(nside)

lon = 0
lat = 0
ctr_pix = hp.ang2pix(nside=nside, theta=lon, phi=lat, lonlat=True)
pix_lon, pix_lat = hp.pix2ang(nside=nside, ipix=ctr_pix, lonlat=True)
print(f'{pix_lon=}, {pix_lat=}')

m = np.load('./sim_30.npy')





