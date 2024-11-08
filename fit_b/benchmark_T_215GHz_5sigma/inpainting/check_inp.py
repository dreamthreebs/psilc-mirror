import numpy as np
import healpy as hp
import pandas as pd
import os,sys
import matplotlib.pyplot as plt

from pathlib import Path
config_dir = Path(__file__).parent.parent
print(f'{config_dir=}')
sys.path.insert(0, str(config_dir))
from config import freq, lmax, nside, beam

flux_idx = 3
rlz_idx = 0
df = pd.read_csv(f'../mask/{freq}.csv')
lon = np.rad2deg(df.at[flux_idx, 'lon'])
lat = np.rad2deg(df.at[flux_idx, 'lat'])

def check_inp():

    input_pcfn = hp.read_map(f'./input/{rlz_idx}.fits')
    output_pcfn = hp.read_map(f'./output/{rlz_idx}.fits')

    hp.gnomview(input_pcfn, rot=[lon, lat, 0])
    hp.gnomview(output_pcfn, rot=[lon, lat, 0])
    plt.show()

def check_slope():
    for rlz_idx in range(200):
        slope = np.load(f'./eblc_slope/{rlz_idx}.npy')
        print(f'{slope=}')

# check_slope()
check_inp()
