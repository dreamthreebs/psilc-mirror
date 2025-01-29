import numpy as np
import healpy as hp
import pandas as pd
import pymaster as nmt
import matplotlib.pyplot as plt
import os,sys

from pathlib import Path
config_dir = Path(__file__).parent.parent
print(f'{config_dir=}')
sys.path.insert(0, str(config_dir))
from config import freq, lmax, nside, beam

threshold = 3

# apo_scale = beam * 8 / 60

df = pd.read_csv(f'../mask/{freq}_after_filter.csv')
ori_mask = np.load('../../../src/mask/north/BINMASKG2048.npy')
ori_apo_mask = np.load('../../../psfit/fitv4/fit_res/2048/ps_mask/new_mask/apo_C1_3_apo_3_apo_3.npy')

rlz_idx = 0
print(f'{rlz_idx=}')
# mask_list = np.load(f'../pcn_after_removal/{threshold}sigma/mask_{rlz_idx}.npy')
# print(f'{mask_list.shape=}')

mask = np.ones_like(ori_mask)

# mask_wym = np.ones_like(ori_mask)
# mask_wym[ori_apo_mask < 1] = 0
# mask_edge = np.zeros_like(ori_mask)
# mask_edge[(ori_apo_mask > 0) & (ori_apo_mask < 1)] = 1

for flux_idx in range(len(df)):
    print(f'{flux_idx=}')
    lon = np.rad2deg(df.at[flux_idx, 'lon'])
    lat = np.rad2deg(df.at[flux_idx, 'lat'])

    ctr_vec = hp.ang2vec(theta=lon, phi=lat, lonlat=True)
    ipix_mask = hp.query_disc(nside=nside, vec=ctr_vec, radius=1.5 * np.deg2rad(beam) / 60)
    mask[ipix_mask] = 0

    # hp.gnomview(ori_mask, rot=[lon, lat, 0], title='before mask', xsize=fig_size)
    # hp.gnomview(mask, rot=[lon, lat, 0], title='after mask', xsize=fig_size)
    # plt.show()

# hp.orthview(mask, rot=[100,50, 0], title='mask', xsize=2000)
# plt.show()

apo_ps_mask = nmt.mask_apodization(mask, aposize=1, apotype='C1') * ori_apo_mask
hp.orthview(apo_ps_mask, rot=[100,50, 0], title='mask', xsize=2000)
plt.show()

# path_ps_mask = Path(f'./3sigma/apo_mask/apo_c1_1')
# path_ps_mask.mkdir(exist_ok=True, parents=True)
# np.save(path_ps_mask / Path(f'{rlz_idx}.npy'), apo_ps_mask)
np.save('./new_mask/apo_ps_mask.npy', apo_ps_mask)










