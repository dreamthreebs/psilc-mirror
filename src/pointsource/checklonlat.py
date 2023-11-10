import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from scipy.io import readsav
from astropy import units as u

data = readsav('/sharefs/alicpt/users/zrzhang/allFreqPSMOutput/skyinbands/AliCPT_uKCMB/30GHz/strongirps_cat_30GHz.sav', python_dict=True, verbose=True)

lon = data['comp']['lon'][0][0][0]
print(f'{lon.shape=}')
lat = data['comp']['lat'][0][0][0]
print(f'{lat.shape=}')

iflux = data['comp']['obs1'][0]['iflux'][0]
print(f'{iflux.shape=}')
inonzero_count = np.count_nonzero(iflux)
print(f'{inonzero_count=}')
qflux = data['comp']['obs1'][0]['qflux'][0]
print(f'{qflux.shape=}')
uflux = data['comp']['obs1'][0]['uflux'][0]
print(f'{uflux.shape=}')

nside = 2048
imap = np.zeros(hp.nside2npix(nside))
qmap = np.zeros(hp.nside2npix(nside))
umap = np.zeros(hp.nside2npix(nside))
print(f'{imap.shape=}')

for i in range(len(lon)):
    pix_index = hp.ang2pix(nside, np.rad2deg(lon[i]), np.rad2deg(lat[i]), lonlat=True)
    imap[pix_index] = iflux[i]
    qmap[pix_index] = qflux[i]
    umap[pix_index] = uflux[i]


imap_out = hp.read_map('/sharefs/alicpt/users/zrzhang/allFreqPSMOutput/skyinbands/AliCPT_uKCMB/30GHz/strongirps_map_30GHz.fits', field=(0,1,2))[0]
qmap_out = hp.read_map('/sharefs/alicpt/users/zrzhang/allFreqPSMOutput/skyinbands/AliCPT_uKCMB/30GHz/strongirps_map_30GHz.fits', field=(0,1,2))[1]
umap_out = hp.read_map('/sharefs/alicpt/users/zrzhang/allFreqPSMOutput/skyinbands/AliCPT_uKCMB/30GHz/strongirps_map_30GHz.fits', field=(0,1,2))[2]


threhold=0
filtered_indices = np.where(imap_out > threhold)
filtered_values = imap_out[filtered_indices]
print(f'2048 {filtered_values.shape=}')

threhold=0
filtered_indices = np.where(imap > threhold)
filtered_values = imap[filtered_indices]
print(f'2048 {filtered_values.shape=}')

imap[imap!=0] = 1
qmap[imap!=0] = 1
umap[imap!=0] = 1
imap_out[imap_out!=0] = 1

hp.mollview(imap, title='my map')
hp.mollview(imap_out, title='out')
hp.mollview(imap-imap_out, title='difference between sav and fits')

# difference_nonzero = np.count_nonzero(imap-imap_out)
difference_nonzero = np.count_nonzero(imap-qmap)
plt.show()
print(f'{difference_nonzero=}')


