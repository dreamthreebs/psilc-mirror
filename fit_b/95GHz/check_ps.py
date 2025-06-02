import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from config import freq, beam, nside

# basic parameters
npix = hp.nside2npix(nside=nside)
nside2pixarea_factor = hp.nside2pixarea(nside=nside)
def check_ps():
    df = pd.read_csv(f'./mask/{freq}.csv')
    print(f'{df=}')
    
    # m_ps = np.load('../data/ps/ps_b.npy')
    m_ps = np.load('./data/ps/ps.npy')
    m_ps_resolved = np.load('./data/ps/resolved_ps.npy')
    # hp.orthview(m_ps[2], rot=[100,50,0])
    # print(f'{len(df)=}')
    
    for flux_idx in range(len(df)):
        print(f'{flux_idx=}, {df.at[flux_idx, "qflux"]}=')
        lon = np.rad2deg(df.at[flux_idx, 'lon'])
        lat = np.rad2deg(df.at[flux_idx, 'lat'])
    
        hp.gnomview(m_ps[1], rot=[lon, lat, 0], reso=1.5, title=f'{flux_idx}')
        hp.gnomview(m_ps_resolved[1], rot=[lon, lat, 0], reso=1.5, title=f'{flux_idx}')
        hp.gnomview(m_ps_resolved[1] - m_ps[1], rot=[lon, lat, 0], reso=1.5, title=f'{flux_idx}')
        plt.show()

check_ps()






