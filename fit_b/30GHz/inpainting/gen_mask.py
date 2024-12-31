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

threshold = 3

df = pd.read_csv(f'../mask/{freq}_after_filter.csv')
ori_mask = np.load('../../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5.npy')
mask = np.ones(hp.nside2npix(nside))

# mask = np.copy(ori_mask)
print(f'{len(df)=}')

for flux_idx in range(len(df)):
    print(f'{flux_idx=}')
    lon = np.rad2deg(df.at[flux_idx, 'lon'])
    lat = np.rad2deg(df.at[flux_idx, 'lat'])

    ctr_vec = hp.ang2vec(theta=lon, phi=lat, lonlat=True)
    ipix_mask = hp.query_disc(nside=nside, vec=ctr_vec, radius=1.8 * np.deg2rad(beam) / 60)
    mask[ipix_mask] = 0

    # fig_size=200
    # # hp.gnomview(ori_mask, rot=[lon, lat, 0], title='before mask', xsize=fig_size)
    # hp.gnomview(mask, rot=[lon, lat, 0], title='after mask', xsize=fig_size)
    # plt.show()

# hp.orthview(mask*ori_mask, rot=[100,50, 0], title='mask', xsize=2000)
# plt.show()

path_mask = Path('./mask')
path_mask.mkdir(exist_ok=True, parents=True)
# hp.write_map(f'./mask/mask_add_edge.fits', mask*ori_mask, overwrite=True)
# hp.write_map(f'./mask/mask.fits', mask*ori_mask, overwrite=True)
hp.write_map(f'./mask/mask_only_edge.fits', ori_mask, overwrite=True)
# hp.write_map(f'./mask/mask1d8.fits', mask*ori_mask, overwrite=True)





