import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from scipy.io import readsav

# data = readsav('/sharefs/alicpt/users/zrzhang/allFreqPSMOutput/skyinbands/AliCPT_uKCMB/270GHz/strongirps_cat_270GHz.sav', python_dict=True, verbose=True)

m = hp.read_map('/sharefs/alicpt/users/zrzhang/allFreqPSMOutput/skyinbands/AliCPT_uKCMB/270GHz/strongradiops_map_270GHz.fits',field=[0,1,2])

# hp.mollview(m[0], title='I', norm='hist')
# hp.mollview(m[1], title='Q', norm='hist')
# hp.mollview(m[2], title='U', norm='hist')
# plt.show()

I = m[0]
Q = m[1]
U = m[2]
print(f'{I.shape=}')
P2 = Q**2 +U**2
threhold = 0
filtered_indices = np.where(I > threhold)
filtered_values = I[filtered_indices]
print(f'{filtered_values.shape}')

hp.gnomview(m[0])

plt.show()





