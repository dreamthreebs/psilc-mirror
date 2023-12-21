import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

nside=2048
m = np.load('./2048.npy')
# m = hp.read_map('/sharefs/alicpt/users/zrzhang/allFreqPSMOutput/observations/AliCPT_uKCMB/40GHz/group1_map_40GHz.fits', field=0)
vec = hp.ang2vec(theta=0, phi=0, lonlat=True)
print(f'{vec=}')
mask = np.zeros(hp.nside2npix(nside))
mask_ipix = hp.query_disc(nside=nside, vec=vec, radius=2.0 * np.deg2rad(63/60))
mask[mask_ipix] = 1
m_mask = m * mask

hp.gnomview(m, rot=[0,0,0])
hp.gnomview(m_mask, rot=[0,0,0])
plt.show()
sum_m = np.sum(m)
print(f'{sum_m=}')
sum_m_mask = np.sum(m_mask)
print(f'{sum_m_mask=}')
