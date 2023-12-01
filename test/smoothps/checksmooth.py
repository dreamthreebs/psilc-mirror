import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from scipy.io import readsav

# data = readsav('/sharefs/alicpt/users/zrzhang/allFreqPSMOutput/observations/AliCPT_uKCMB')
irps = hp.read_map('/sharefs/alicpt/users/zrzhang/allFreqPSMOutput/skyinbands/AliCPT_uKCMB/40GHz/strongirps_map_40GHz.fits',field=0)
# print(f'')
radiops = hp.read_map('/sharefs/alicpt/users/zrzhang/allFreqPSMOutput/skyinbands/AliCPT_uKCMB/40GHz/strongradiops_map_40GHz.fits', field=0)

ps = irps + radiops

# hp.mollview(ps, norm='hist')
# plt.show()

sm_ps = hp.smoothing(ps, lmax=6145, fwhm=np.deg2rad(63)/60, use_pixel_weights=True, iter=10)
# sm_ps = hp.smoothing(ps, lmax=2145, fwhm=np.deg2rad(63)/60, iter=3)

true = hp.read_map('/sharefs/alicpt/users/zrzhang/allFreqPSMOutput/observations/AliCPT_uKCMB/40GHz/group3_map_40GHz.fits', field=0)

diff_m = true - sm_ps
hp.mollview(diff_m, norm='hist')
hp.mollview(true, norm='hist')
hp.mollview(sm_ps, norm='hist')
plt.show()
