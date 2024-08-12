import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

# basic parameters
nside = 2048
beam = 11
npix = hp.nside2npix(nside=nside)
nside2pixarea_factor = hp.nside2pixarea(nside=nside)
freq = 215
df = pd.read_csv('../mask/215.csv')
print(f'{df=}')

m_ps = np.load('../data/ps/ps_b.npy')

for flux_idx in range(20):
    print(f'{flux_idx=}')
    lon = np.rad2deg(df.at[flux_idx, 'lon'])
    lat = np.rad2deg(df.at[flux_idx, 'lat'])

    hp.gnomview(m_ps, rot=[lon, lat, 0], reso=1.5, title=f'{flux_idx}')
    plt.show()





