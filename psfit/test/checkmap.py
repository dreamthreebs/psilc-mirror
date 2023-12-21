import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt


m = np.load('../../FGSim/PSNOISE/2048/40.npy')[0]
m1 = hp.read_map('/sharefs/alicpt/users/zrzhang/allFreqPSMOutput/skyinbands/AliCPT_uKCMB/40GHz/strongradiops_map_40GHz.fits', field=0)
df = pd.read_csv('../../test/ps_sort/sort_by_iflux/40.csv')

lon = df.at[44,'lon']
lat = df.at[44,'lat']

hp.gnomview(m, rot=[np.rad2deg(lon),np.rad2deg(lat),0], reso=0.5, xsize=100)
hp.gnomview(m1, rot=[np.rad2deg(lon),np.rad2deg(lat),0], reso=0.5, xsize=100)
plt.show()
