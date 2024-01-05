import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt

m = hp.read_map('/sharefs/alicpt/users/zrzhang/allFreqPSMOutput/observations/AliCPT_uKCMB/40GHz/group4_map_40GHz.fits', field=0)
m1 = hp.read_map('/sharefs/alicpt/users/zrzhang/allFreqPSMOutput/observations/AliCPT_uKCMB/40GHz/group1_map_40GHz.fits', field=0)
df = pd.read_csv('../../psfit/partial_sky_ps/ps_with_nearby/40.csv')

idx = 30
lon = df.at[idx,'lon']
lat = df.at[idx,'lat']

# hp.mollview(m, norm='hist')
hp.gnomview(m1, rot=[np.rad2deg(lon), np.rad2deg(lat), 0], title='s,d,f,sp,co', reso=1.5, xsize=90, ysize=90)
hp.gnomview(m, rot=[np.rad2deg(lon), np.rad2deg(lat), 0], title='all', reso=1.5, xsize=90, ysize=90)
plt.show()


