import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

nside = 2048
irps = hp.read_map('/sharefs/alicpt/users/zrzhang/allFreqPSMOutput/skyinbands/AliCPT_uKCMB/40GHz/strongirps_map_40GHz.fits', field=0)
radiops = hp.read_map('/sharefs/alicpt/users/zrzhang/allFreqPSMOutput/skyinbands/AliCPT_uKCMB/40GHz/strongradiops_map_40GHz.fits', field=0)
ps = irps + radiops

mask = np.load('../../src/mask/north/BINMASKG2048.npy')
full_n_ps = np.count_nonzero(ps)
north_m = mask * ps
# north_m[north_m > 0] = 1e7
north_m = hp.smoothing(north_m, lmax=500, fwhm=np.deg2rad(63)/60)
north_n_ps = np.count_nonzero(north_m)
print(f'{full_n_ps=}')
print(f'{north_n_ps=}')

north_ps_npix = np.nonzero(north_m)

# for i in range(north_n_ps):
#     lon, lat = hp.pix2ang(nside=nside, ipix=north_ps_npix[0][i], lonlat=True)
#     hp.gnomview(north_m, rot=[lon, lat, 0] )
#     plt.show()

north_m[mask<=0] = 2
# north_m[north_m > 0] = 1
# hp.mollview(north_m, rot=[100,50,0],  xsize=5000)
hp.orthview(hp.ma(north_m, badval=2), rot=[100,50,0],  xsize=5000, half_sky=True, badcolor='white')
plt.show()


