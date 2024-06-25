import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from pathlib import Path

lmax = 2000
nside = 2048
npix = hp.nside2npix(nside)

beam = 11
sigma = np.deg2rad(beam) / 60 / (np.sqrt(8 * np.log(2)))

m_delta = np.zeros((3, npix))

lon = 0
lat = 0
ipix_ctr = hp.ang2pix(theta=lon, phi=lat, lonlat=True, nside=nside)
pix_lon, pix_lat = hp.pix2ang(ipix=ipix_ctr, nside=nside, lonlat=True)
print(f"{pix_lon=}, {pix_lat=}")

flux_density_i = 1e3
flux_density_q = 5e2
flux_density_u = 5e2

flux_density_q = 5e2
flux_density_u = -2.5e2

m_delta[0,ipix_ctr] = flux_density_i
m_delta[1,ipix_ctr] = flux_density_q
m_delta[2,ipix_ctr] = flux_density_u
sm_m = hp.smoothing(m_delta, fwhm=np.deg2rad(beam)/60, lmax=lmax)

hp.gnomview(sm_m[0], rot=[lon, lat, 0], xsize=100, title='T')
hp.gnomview(sm_m[1], rot=[lon, lat, 0], xsize=100, title='Q')
hp.gnomview(sm_m[2], rot=[lon, lat, 0], xsize=100, title='U')
plt.show()

nstd = np.load('../../FGSim/NSTDNORTH/2048/215.npy')

noise = nstd * np.random.normal(loc=0, scale=1, size=(3,npix))
m = noise + sm_m

# np.save('./data/sim.npy', m)
# np.save('./data/ps.npy', sm_m)
# np.save('./data/noise.npy', noise)

# np.save('./data/sim.npy', m)
np.save('../test_ps_on_b/ps.npy', sm_m)
# np.save('./data/noise.npy', noise)





