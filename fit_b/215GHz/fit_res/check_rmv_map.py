import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt
import os, sys

from pathlib import Path
config_dir = Path(__file__).parent.parent
print(f'{config_dir=}')
sys.path.insert(0, str(config_dir))
from config import freq, lmax, nside, beam

df = pd.read_csv(f'../mask/{freq}_after_filter.csv')
m = np.load('./std/3sigma/map_u_0.npy')

for flux_idx in range(3):
    lon = np.rad2deg(df.at[flux_idx, 'lon'])
    lat = np.rad2deg(df.at[flux_idx, 'lat'])

    hp.gnomview(m, rot=[lon, lat, 0])
    plt.show()

