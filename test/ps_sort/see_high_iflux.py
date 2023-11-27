import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pandas as pd

df30 = pd.read_csv('./sort_by_iflux/30.csv')

n_ps = len(df30)
frac_threshold = 0.1
n_effps = frac_threshold * n_ps
lon = df30.loc[:, 'lon'].to_numpy()
lat = df30.loc[:, 'lat'].to_numpy()
iflux = df30.loc[:, 'iflux'].to_numpy()

# ps_m = hp.read_map('/sharefs/alicpt/users/zrzhang/allFreqPSMOutput/skyinbands/AliCPT_uKCMB/30GHz/strongradiops_map_30GHz.fits', field=0)
# hp.gnomview(ps_m, rot=[np.rad2deg(lon), np.rad2deg(lat), 0] )
# plt.show()

nside = 2048
npix = hp.nside2npix(nside)
m = np.zeros(npix)
for i in range(len(df30)):
    print(f'{i=}')
    ipix = hp.ang2pix(nside=nside, theta=np.rad2deg(lon[i]), phi=np.rad2deg(lat[i]), lonlat=True)
    m[ipix] = m[ipix] + iflux[i]

hp.gnomview(m, rot=[np.rad2deg(lon[0]), np.rad2deg(lat[0]), 0] )
plt.show()

