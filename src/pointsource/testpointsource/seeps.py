import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

strongps = hp.read_map('/sharefs/alicpt/users/zrzhang/allFreqPSMOutput/observations/AliCPT_uKCMB/270GHz/group3_map_270GHz.fits', field=(0,1,2))
faintps = hp.read_map('/sharefs/alicpt/users/zrzhang/allFreqPSMOutput/observations/AliCPT_uKCMB/270GHz/group2_map_270GHz.fits', field=(0,1,2))

# for i, m in enumerate('TQU'):
#     hp.mollview(strongps[i], title=f'strong ps {m} map', cmap='Blues', norm='hist')
#     # hp.mollview(np.abs(faintps[i]), title=f'faint ps {m} map', norm='log', min=1e-5, cmap='Blues')
#     plt.show()

strongps_P2 = strongps[1]**2 + strongps[2]**2
faintps_P2 = faintps[1]**2 + faintps[2]**2
threhold = 0
filtered_indices = np.where(strongps_P2 > threhold)
filtered_values = strongps_P2[filtered_indices]
print(f'{filtered_values.shape}')

hp.mollview(strongps_P2, norm='hist', title='strong ps P2')
hp.mollview(faintps_P2, norm='log', title='faint ps P2', min=1e-4)
plt.show()


