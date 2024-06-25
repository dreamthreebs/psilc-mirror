import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

nstd = 0.1
nside = 2048

ctr_ori_lon = 0
ctr_ori_lat = 0

ipix_ctr = hp.ang2pix(theta=ctr_ori_lon, phi=ctr_ori_lat, lonlat=True, nside=nside)
ctr_theta, ctr_phi = hp.pix2ang(nside=nside, ipix=ipix_ctr)
ctr_vec = np.asarray(hp.pix2vec(nside=nside, ipix=ipix_ctr))
ctr_lon, ctr_lat = hp.pix2ang(nside=nside, ipix=ipix_ctr, lonlat=True)
print(f'{ctr_theta=}, {ctr_phi=}, {ctr_vec=}')

ps = np.load('./m_p_b.npy')

print(f'{ps.size=}')
print(f'{hp.nside2npix(2048)}')

noise = nstd * np.random.normal(loc=0, scale=1, size=ps.size)
hp.mollview(noise)
plt.show()

m_pn_b = noise + ps
hp.gnomview(m_pn_b, rot=[ctr_lon, ctr_lat, 0])
plt.show()
np.save('m_pn_b.npy', m_pn_b)


