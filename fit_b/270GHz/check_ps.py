import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from config import nside, beam, freq
from fit_qu_no_const import FitPolPS

# basic parameters
npix = hp.nside2npix(nside=nside)
nside2pixarea_factor = hp.nside2pixarea(nside=nside)
df = pd.read_csv(f'./mask/{freq}.csv')
print(f'{df=}')

# m_ps = np.load('../data/ps/ps_b.npy')
m_ps = np.load('./data/ps/ps.npy')
hp.orthview(m_ps[1], rot=[100,50,0])
print(f'{len(df)=}')

sigma = np.deg2rad(beam) / 60 / (np.sqrt(8 * np.log(2)))

for flux_idx in range(len(df)):
    print(f'{flux_idx=}')

    print(f'{df.at[flux_idx, "qflux"]=}')
    Tpeak = df.at[flux_idx, "qflux"] * FitPolPS.mJy_to_uKCMB(intensity_mJy=1, frequency_GHz=freq) / (2*np.pi*sigma**2)
    print(f'{Tpeak=}')

    lon = np.rad2deg(df.at[flux_idx, 'lon'])
    lat = np.rad2deg(df.at[flux_idx, 'lat'])

    hp.gnomview(m_ps[1], rot=[lon, lat, 0], reso=1.5, title=f'{flux_idx}')
    plt.show()









