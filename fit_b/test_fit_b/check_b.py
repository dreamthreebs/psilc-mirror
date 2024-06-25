import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

lmax = 2000
nside = 2048
npix = hp.nside2npix(nside)

beam = 11

# m = np.load('./data/ps.npy')
# alm_t, alm_e, alm_b = hp.map2alm(m, lmax=lmax)
# m_e = hp.alm2map(alm_e, nside=nside)
# m_b = hp.alm2map(alm_b, nside=nside)

# np.save('./data/ps_e.npy', m_e)
# np.save('./data/ps_b.npy', m_b)

m_e = np.load('./data/ps_e.npy')
m_b = np.load('./data/ps_b.npy')

lon = 0
lat = 0
ipix_ctr = hp.ang2pix(theta=lon, phi=lat, lonlat=True, nside=nside)
pix_lon, pix_lat = hp.pix2ang(ipix=ipix_ctr, nside=nside, lonlat=True)
ctr_vec = hp.pix2vec(nside=nside, ipix=ipix_ctr)
print(f"{pix_lon=}, {pix_lat=}")

hp.gnomview(m_e, rot=[pix_lon, pix_lat, 0], title='E', xsize=100)
hp.gnomview(m_b, rot=[pix_lon, pix_lat, 0], title='B', xsize=100)
plt.show()

ipix_disc = hp.query_disc(nside=nside, vec=ctr_vec, radius=3 * np.deg2rad(beam)/60)
mask = np.ones(npix)
mask[ipix_disc] = 0

hp.gnomview(m_b * mask, rot=[pix_lon, pix_lat, 0], title='masked B', xsize=100)
plt.show()

