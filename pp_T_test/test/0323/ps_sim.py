import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

nside = 2048
lmax = 2000
npix = hp.nside2npix(nside)
m = np.zeros(npix)

lon = 0
lat = 0
ctr_pix = hp.ang2pix(nside=nside, theta=lon, phi=lat, lonlat=True)
pix_lon, pix_lat = hp.pix2ang(nside=nside, ipix=ctr_pix, lonlat=True)
print(f'{pix_lon=}, {pix_lat=}')

m[ctr_pix] = 3000

sm = hp.smoothing(m, fwhm=np.deg2rad(30)/60, lmax=lmax)
sm1 = hp.smoothing(m, fwhm=np.deg2rad(17)/60, lmax=lmax)
np.save('ps_30.npy', sm)
np.save('ps_17.npy', sm1)

hp.gnomview(sm, rot=[pix_lon, pix_lat, 0], title='30')
hp.gnomview(sm1, rot=[pix_lon, pix_lat, 0], title='17')
plt.show()

# disc_vec = hp.ang2vec(theta=lon, phi=lat, lonlat=True)
# hp.query_disc(nside=nside, vec=disc_vec, radius=1.5)


