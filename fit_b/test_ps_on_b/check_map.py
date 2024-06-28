import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

lmax = 2000
nside = 2048
npix = hp.nside2npix(nside)
ps = np.load('./data/ps.npy')
beam = 11

ctr_ori_lon = 0
ctr_ori_lat = 0

ipix_ctr = hp.ang2pix(theta=ctr_ori_lon, phi=ctr_ori_lat, lonlat=True, nside=nside)
ctr_theta, ctr_phi = hp.pix2ang(nside=nside, ipix=ipix_ctr)
ctr_vec = np.asarray(hp.pix2vec(nside=nside, ipix=ipix_ctr))
ctr_lon, ctr_lat = hp.pix2ang(nside=nside, ipix=ipix_ctr, lonlat=True)
print(f'{ctr_theta=}, {ctr_phi=}, {ctr_vec=}')

m = np.load('./m_qu_cn.npy')
m1 = np.load('./m_cn_b_1.npy')

hp.gnomview(m, rot=[ctr_lon, ctr_lat, 0], xsize=50, title='qu noise')
hp.gnomview(m1, rot=[ctr_lon, ctr_lat, 0], xsize=50, title='b noise')
plt.show()





