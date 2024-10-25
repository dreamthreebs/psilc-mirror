import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

# basic parameters
nside = 2048
beam = 30
npix = hp.nside2npix(nside=nside)
nside2pixarea_factor = hp.nside2pixarea(nside=nside)
freq = 95
df = pd.read_csv(f'./mask/{freq}.csv')
print(f'{df=}')

# m_ps = np.load('../data/ps/ps_b.npy')
m_ps = np.load('./data/ps/ps.npy')
hp.orthview(m_ps[1], rot=[100,50,0])
print(f'{len(df)=}')

for flux_idx in range(len(df)):
    print(f'{flux_idx=}')
    lon = np.rad2deg(df.at[flux_idx, 'lon'])
    lat = np.rad2deg(df.at[flux_idx, 'lat'])

    hp.gnomview(m_ps[1], rot=[lon, lat, 0], reso=1.5, title=f'{flux_idx}')
    plt.show()








