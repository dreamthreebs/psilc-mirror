import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

def gen_mask_pcn(ori_mask, nside, radius_factor, beam, df):
    for rlz_idx in range(100):
        print(f'{rlz_idx=}')
        mask_list = np.load(f'./ps_cmb_noise_residual/2sigma/mask{rlz_idx}.npy')
        print(f'{mask_list=}')

        # m = np.load('./ps_cmb_noise_residual/map0.npy')
        # m = np.load('../../../../fitdata/synthesis_data/2048/PSCMBNOISE/155/0.npy')[0]

        # hp.orthview(m*mask, rot=[100,50,0], half_sky=True)
        # plt.show()

        mask = np.copy(ori_mask)
        for flux_idx in mask_list:
            print(f'{flux_idx=}')
            pcn_fit_lon = np.rad2deg(df.at[flux_idx, 'lon'])
            pcn_fit_lat = np.rad2deg(df.at[flux_idx, 'lat'])

            ctr0_pix = hp.ang2pix(nside=nside, theta=pcn_fit_lon, phi=pcn_fit_lat, lonlat=True)
            ctr0_vec = np.array(hp.pix2vec(nside=nside, ipix=ctr0_pix)).astype(np.float64)

            ipix_mask = hp.query_disc(nside=nside, vec=ctr0_vec, radius=radius_factor * np.deg2rad(beam) / 60)
            mask[ipix_mask] = 0

        # hp.orthview(m*ori_mask, rot=[100,50,0], half_sky=True, title='origin')
        # hp.orthview(m*mask, rot=[100,50,0], half_sky=True,title='masked')
        # plt.show()

        mask_path = Path(f'./INPAINT/mask/pcn/2sigma')
        mask_path.mkdir(parents=True, exist_ok=True)
        hp.write_map(mask_path / Path(f'{rlz_idx}.fits'), mask, overwrite=True)

def gen_mask_pcfn(ori_mask, nside, radius_factor, beam, df):
    for rlz_idx in range(100):
        print(f'{rlz_idx=}')
        mask_list = np.load(f'./ps_cmb_fg_noise_residual/2sigma/mask{rlz_idx}.npy')
        print(f'{mask_list=}')

        # m = np.load('./ps_cmb_fg_noise_residual/2sigma/map0.npy')
        # m = np.load(f'../../../../fitdata/synthesis_data/2048/PSCMBFGNOISE/{freq}/0.npy')[0]

        # hp.orthview(m*mask, rot=[100,50,0], half_sky=True)
        # plt.show()

        mask = np.copy(ori_mask)
        for flux_idx in mask_list:
            print(f'{flux_idx=}')
            pcfn_fit_lon = np.rad2deg(df.at[flux_idx, 'lon'])
            pcfn_fit_lat = np.rad2deg(df.at[flux_idx, 'lat'])

            ctr0_pix = hp.ang2pix(nside=nside, theta=pcfn_fit_lon, phi=pcfn_fit_lat, lonlat=True)
            ctr0_vec = np.array(hp.pix2vec(nside=nside, ipix=ctr0_pix)).astype(np.float64)

            ipix_mask = hp.query_disc(nside=nside, vec=ctr0_vec, radius=radius_factor * np.deg2rad(beam) / 60)
            mask[ipix_mask] = 0

        # hp.orthview(m*ori_mask, rot=[100,50,0], half_sky=True, title='origin', xsize=2000)
        # hp.orthview(m*mask, rot=[100,50,0], half_sky=True,title='masked', xsize=2000)
        # plt.show()

        mask_path = Path(f'./INPAINT/mask/pcfn/2sigma')
        mask_path.mkdir(parents=True, exist_ok=True)
        output_path = Path(f'./INPAINT/output/pcfn/2sigma')
        output_path.mkdir(parents=True, exist_ok=True)

        hp.write_map(mask_path / Path(f'{rlz_idx}.fits'), mask, overwrite=True)



if __name__ == "__main__":
    freq = 30
    df = pd.read_csv(f'../../../mask/mask_csv/{freq}.csv')
    mask = np.load('../../../../src/mask/north/BINMASKG2048.npy')
    nside = 2048
    radius_factor = 1.5
    beam = 67
    # gen_mask_pcn(mask, nside=nside, radius_factor=radius_factor, beam=beam, df=df)
    gen_mask_pcfn(mask, nside=nside, radius_factor=radius_factor, beam=beam, df=df)



