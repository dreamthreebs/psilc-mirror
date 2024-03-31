import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

nside = 2048
lmax = 2000
npix = hp.nside2npix(nside)

lon = 0
lat = 0
ctr_pix = hp.ang2pix(nside=nside, theta=lon, phi=lat, lonlat=True)
pix_lon, pix_lat = hp.pix2ang(nside=nside, ipix=ctr_pix, lonlat=True)
print(f'{pix_lon=}, {pix_lat=}')

sim_30 = np.load('../../../fitdata/synthesis_data/2048/CMBNOISE/95/0.npy')[0] + np.load('./ps_30.npy')
sim_17 = np.load('../../../fitdata/synthesis_data/2048/CMBNOISE/155/0.npy')[0] + np.load('./ps_17.npy')
np.save('sim_30.npy', sim_30)
np.save('sim_17.npy', sim_17)

hp.gnomview(sim_30, rot=[pix_lon, pix_lat, 0], title='30')
hp.gnomview(sim_17, rot=[pix_lon, pix_lat, 0], title='17')
plt.show()




