import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import readsav

df = pd.read_csv('../ps_sort/sort_by_iflux/40.csv')

lon = df.loc[:, 'lon']
print(f'{lon.shape=}')
lat = df.loc[:, 'lat']
iflux = df.loc[:, 'iflux']

radiops = hp.read_map('/sharefs/alicpt/users/zrzhang/allFreqPSMOutput/skyinbands/AliCPT_uKCMB/40GHz/strongradiops_map_40GHz.fits', field=0)
irps = hp.read_map('/sharefs/alicpt/users/zrzhang/allFreqPSMOutput/skyinbands/AliCPT_uKCMB/40GHz/strongirps_map_40GHz.fits', field=0)
nside = 2048
ps = radiops + irps
# ps[ps>0] = 1

# m = np.zeros(hp.nside2npix(nside))
# ipix = hp.ang2pix(nside=nside, theta=np.rad2deg(lon), phi=np.rad2deg(lat), lonlat=True)

m = np.load('../../FGSim/PSNOISE/40.npy')[0]
hp.gnomview(ps, xsize=200,reso=0.5, rot=[np.rad2deg(lon[44]), np.rad2deg(lat[44]), 0], title=f'40')
hp.projscatter(theta=np.rad2deg(lon[44]), phi=np.rad2deg(lat[44]), lonlat=True)
plt.show()

# for i in range(len(lon)):
#     hp.gnomview(ps, rot=[np.rad2deg(lon[i]), np.rad2deg(lat[i]), 0], title=f'{i}')
#     hp.projscatter(theta=np.rad2deg(lon[i]), phi=np.rad2deg(lat[i]), lonlat=True)
#     plt.show()




