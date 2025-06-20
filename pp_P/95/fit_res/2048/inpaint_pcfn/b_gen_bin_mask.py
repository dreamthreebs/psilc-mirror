import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt

threshold = 3
freq = 95
beam = 30
nside = 2048

df = pd.read_csv(f'../../../../mask/mask_csv/{freq}.csv')
ori_mask = np.load('../../../../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5.npy')

for rlz_idx in range(100):
    print(f'{rlz_idx=}')
    mask_list = np.load(f'../pcfn_after_removal/{threshold}sigma/mask_{rlz_idx}.npy')
    print(f'{mask_list.shape=}')

    mask = np.copy(ori_mask)

    for flux_idx in mask_list:
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

    hp.write_map(f'./{threshold}sigma/bin_mask/{rlz_idx}.fits', mask, overwrite=True)



