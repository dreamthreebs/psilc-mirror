import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt
import os,sys

from pathlib import Path
from eblc_base_slope import EBLeakageCorrection
config_dir = Path(__file__).parent.parent
print(f'{config_dir=}')
sys.path.insert(0, str(config_dir))
from config import freq, lmax, nside, beam

df = pd.read_csv(f'../mask/{freq}.csv')

def check_inp():
    input_pcfn = hp.read_map('./input_m2/0.fits')
    output_pcfn = hp.read_map('./output_m2/0.fits')

    input_n = hp.read_map('./input_m2_n/0.fits')
    output_n = hp.read_map('./output_m2_n/0.fits')

    flux_idx = 5
    lon = np.rad2deg(df.at[flux_idx, 'lon'])
    lat = np.rad2deg(df.at[flux_idx, 'lat'])

    hp.gnomview(input_pcfn, rot=[lon, lat, 0], title='input pcfn')
    hp.gnomview(input_n, rot=[lon, lat, 0], title='input n')
    hp.gnomview(output_pcfn, rot=[lon, lat, 0], title='output pcfn')
    hp.gnomview(output_n, rot=[lon, lat, 0], title='output n')
    plt.show()

check_inp()



