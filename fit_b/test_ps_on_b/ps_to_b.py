import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

lmax = 2000
nside = 2048
m = np.load('./ps.npy')
beam = 11

ctr_ori_lon = 0
ctr_ori_lat = 0

ipix_ctr = hp.ang2pix(theta=ctr_ori_lon, phi=ctr_ori_lat, lonlat=True, nside=nside)
ctr_theta, ctr_phi = hp.pix2ang(nside=nside, ipix=ipix_ctr)
ctr_vec = np.asarray(hp.pix2vec(nside=nside, ipix=ipix_ctr))
ctr_lon, ctr_lat = hp.pix2ang(nside=nside, ipix=ipix_ctr, lonlat=True)
print(f'{ctr_theta=}, {ctr_phi=}, {ctr_vec=}')

m_b = hp.alm2map(hp.map2alm(m, lmax=lmax)[2], nside=nside)

hp.gnomview(m_b, rot=[ctr_lon, ctr_lat, 0])
plt.show()

ipix_disc = hp.query_disc(nside=nside, vec=ctr_vec, radius=3*np.deg2rad(beam)/60)
sum_m = np.sum(np.abs(m_b[ipix_disc].copy()))
print(f'{sum_m=}')
np.save('./sum.npy', sum_m)
np.save('./m_b.npy', m_b)


