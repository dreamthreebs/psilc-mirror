import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

m = hp.read_map('/sharefs/alicpt/users/zrzhang/allFreqPSMOutput/observations/AliCPT_uKCMB/30GHz/group3_map_30GHz.fits', field=(0,1,2))
hp.mollview(m[0], norm='hist')
plt.show()
