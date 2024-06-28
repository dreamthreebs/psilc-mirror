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

nstd = 0.1
noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))
print(f"{np.std(noise[1])=}")

cmb_iqu = np.load('../../fitdata/2048/CMB/215/1.npy')

pcn = noise + ps + cmb_iqu
cn = cmb_iqu + noise

n_b = hp.alm2map(hp.map2alm(noise)[2], nside=nside)
print(f"{np.std(n_b)=}")
pcn_b = hp.alm2map(hp.map2alm(pcn)[2], nside=nside)
cn_b = hp.alm2map(hp.map2alm(cn)[2], nside=nside)



hp.gnomview(cn_b, rot=[ctr_lon, ctr_lat, 0], title='cn b')
hp.gnomview(pcn_b, rot=[ctr_lon, ctr_lat, 0], title='pcn b')
hp.gnomview(n_b, rot=[ctr_lon, ctr_lat, 0], title='n b')
plt.show()

np.save('./m_qu_pcn.npy', pcn_b)
np.save('./m_qu_cn.npy', cn_b)
np.save('./m_qu_n.npy', n_b)

