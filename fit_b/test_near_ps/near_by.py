import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from pathlib import Path

beam = 11
nside = 2048
lmax = 3*nside -1
npix = hp.nside2npix(nside)

m_delta = np.zeros((3,npix))

lon1 = 0
lat1 = 0
ipix_ctr1 = hp.ang2pix(theta=lon1, phi=lat1, lonlat=True, nside=nside)
pix_lon1, pix_lat1 = hp.pix2ang(ipix=ipix_ctr1, nside=nside, lonlat=True)

flux_q1 = 5e2
flux_u1 = -2.5e2
path_ps = Path('./data/ps')
path_ps.mkdir(exist_ok=True, parents=True)

distance_factor = 0.5
print(f'{distance_factor=}')

lon2 = distance_factor * beam / 60
lat2 = 0
ipix_ctr2 = hp.ang2pix(theta=lon2, phi=lat2, lonlat=True, nside=nside)
pix_lon2, pix_lat2 = hp.pix2ang(ipix=ipix_ctr2, nside=nside, lonlat=True)

flux_q2 = 5e2
flux_u2 = 5e2

m_delta[1,ipix_ctr1] = flux_q1
m_delta[2,ipix_ctr1] = flux_u1
m_delta[1,ipix_ctr2] = flux_q2
m_delta[2,ipix_ctr2] = flux_u2
sm_m = hp.smoothing(m_delta, fwhm=np.deg2rad(beam)/60)
m = hp.alm2map(hp.map2alm(sm_m)[2], nside=nside)

angdist = hp.rotator.angdist(dir1=(lon1,lat1), dir2=(lon2,lat2), lonlat=True)
print(f'{np.rad2deg(angdist)=}')

np.save(path_ps / Path(f'ps_map_{distance_factor}.npy'), m)

# nstd = 0.1
# np.random.seed()
# noise = nstd * np.random.normal(loc=0, scale=1, size=(3,npix))
# m = sm_m + noise

# hp.gnomview(sm_m[1], rot=[pix_lon1, pix_lat1, 0])
# hp.gnomview(sm_m[2], rot=[pix_lon1, pix_lat1, 0])
# plt.show()



