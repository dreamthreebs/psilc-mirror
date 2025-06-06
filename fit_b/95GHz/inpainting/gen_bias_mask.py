import numpy as np
import healpy as hp
import pandas as pd
import os,sys
import matplotlib.pyplot as plt
import pymaster as nmt

from pathlib import Path
config_dir = Path(__file__).parent.parent
print(f'{config_dir=}')
sys.path.insert(0, str(config_dir))
from config import freq, lmax, nside, beam

threshold = 3

df = pd.read_csv(f'../mask/{freq}_after_filter.csv')
# ori_mask = np.load('../../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5.npy')
ori_mask = np.load('../../../src/mask/north/BINMASKG2048.npy')

# mask = np.copy(ori_mask)
print(f'{len(df)=}')

def gen_inp_bias_mask():
    mask = np.ones(hp.nside2npix(nside))
    for flux_idx in range(len(df)):
        print(f'{flux_idx=}')
        lon = np.rad2deg(df.at[flux_idx, 'lon'])
        lat = np.rad2deg(df.at[flux_idx, 'lat'])
    
        ctr_vec = hp.ang2vec(theta=lon, phi=lat, lonlat=True)
        ipix_mask = hp.query_disc(nside=nside, vec=ctr_vec, radius=2.5 * np.deg2rad(beam) / 60)
        mask[ipix_mask] = 0
    
        # fig_size=200
        # # hp.gnomview(ori_mask, rot=[lon, lat, 0], title='before mask', xsize=fig_size)
        # hp.gnomview(mask, rot=[lon, lat, 0], title='after mask', xsize=fig_size)
        # plt.show()
    
    apo_mask = nmt.mask_apodization(mask_in=mask, aposize=1)
    # hp.orthview(mask*ori_mask, rot=[100,50, 0], title='mask', xsize=2000)
    # plt.show()
    
    path_mask = Path('./mask_bias')
    path_mask.mkdir(exist_ok=True, parents=True)
    np.load(f'./mask_bias/inp.npy', 1 - apo_mask)

def check_inp_bias_mask():
    apo_mask = np.load('./mask_bias/inp.npy')

    hp.orthview(apo_mask, rot=[100,50,0], half_sky=True, title='apo_mask')
    plt.show()


def gen_masking_bias_mask():
    mask = np.ones(hp.nside2npix(nside))
    for flux_idx in range(len(df)):
        print(f'{flux_idx=}')
        lon = np.rad2deg(df.at[flux_idx, 'lon'])
        lat = np.rad2deg(df.at[flux_idx, 'lat'])
    
        ctr_vec = hp.ang2vec(theta=lon, phi=lat, lonlat=True)
        ipix_mask = hp.query_disc(nside=nside, vec=ctr_vec, radius=2.5 * np.deg2rad(beam) / 60)
        mask[ipix_mask] = 0
    
        # fig_size=200
        # # hp.gnomview(ori_mask, rot=[lon, lat, 0], title='before mask', xsize=fig_size)
        # hp.gnomview(mask, rot=[lon, lat, 0], title='after mask', xsize=fig_size)
        # plt.show()
    
    apo_mask = nmt.mask_apodization(mask_in=mask, aposize=3)
    hp.orthview(mask*ori_mask, rot=[100,50, 0], title='mask', xsize=2000)
    plt.show()
    
    path_mask = Path('./mask_bias')
    path_mask.mkdir(exist_ok=True, parents=True)
    np.save(f'./mask_bias/C1_3.npy', 1 - apo_mask)

def check_masking_bias_mask():
    apo_mask = np.load('./mask_bias/C1_2.npy')
    apo_mask_inp = np.load('./mask_bias/inp.npy')
    apo_mask_ps_mask = np.load(f"./new_mask/apo_ps_mask.npy")

    hp.orthview(apo_mask, rot=[100,50,0], half_sky=True, title='apo_mask')
    hp.orthview(apo_mask_inp, rot=[100,50,0], half_sky=True, title='apo_mask inp')
    hp.orthview(1 - apo_mask_ps_mask - apo_mask, rot=[100,50,0], half_sky=True, title='ps mask')
    plt.show()



# gen_inp_bias_mask()
gen_masking_bias_mask()
# check_inp_bias_mask()
# check_masking_bias_mask()




