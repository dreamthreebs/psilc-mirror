import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from scipy.io import readsav
from astropy import units as u

data = readsav('/sharefs/alicpt/users/zrzhang/allFreqPSMOutput/skyinbands/AliCPT_uKCMB/30GHz/strongradiops_cat_30GHz.sav', python_dict=True, verbose=True)

lon = data['comp']['lon'][0][0][0]
print(f'{lon.shape=}')
lat = data['comp']['lat'][0][0][0]
print(f'{lat.shape=}')

nside = 2048
m = np.ones(hp.nside2npix(nside))

for i in range(len(lon)):
    vec = hp.ang2vec(np.rad2deg(lon[i]), np.rad2deg(lat[i],), lonlat=True)
    ipix = hp.query_disc(nside, vec, radius=np.deg2rad(40)/60)
    # print(f'{ipix=}')
    m[ipix] = 0

hp.mollview(m)
plt.show()

