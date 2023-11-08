import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
from pathlib import Path

def bl_curl_creater(df, lmax):
    bl_curl_list = []
    n_freq = len(df)
    for i in range(n_freq):
        freq = df.at[i,'freq']
        beam = df.at[i,'beam'] # arcmin
        print(f'{freq=},{beam=}')
        bl_curl = hp.gauss_beam(fwhm=np.deg2rad(beam/60), lmax=lmax, pol=True)[:,2]
        bl_curl_list.append(bl_curl)
    bl_curl_arr = np.array(bl_curl_list)
    print(f'{bl_curl_arr.shape}')
    return bl_curl_arr

def smooth_b(df, bl_std_curl, bl_curl, m_list, bin_mask, apo_mask, lmax, nside, save_path, smooth_scale=30):
    smooth_list = []
    for index, m_pos in enumerate(m_list):
        m = np.load(m_pos) * hp.smoothing(bin_mask, fwhm=np.deg2rad(smooth_scale)/60)

        alm_ori = hp.map2alm(m, lmax=lmax)
        alm_base = hp.almxfl(alm_ori, bl_std_curl / bl_curl[index])

        sm_b = hp.alm2map(alm_base, nside=nside) * bin_mask
        freq = df.at[index, 'freq']

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        smooth_list.append(sm_b)
    np.save(f'{save_path}/data.npy', np.array(smooth_list))


if __name__=='__main__':
    nside = 512
    lmax = 500
    beam_base = 9 # arcmin
    df = pd.read_csv('../../FGSim/FreqBand')
    n_freq = len(df)

    bl_std_curl = hp.gauss_beam(fwhm=np.deg2rad(beam_base/60), lmax=lmax, pol=True)[:,2]
    bl_curl = bl_curl_creater(df=df, lmax=lmax)

    cmb_list = sorted(glob.glob(f'../eblc/eblc_data/sim/*.npy'), key=lambda x: int(Path(x).stem))
    bin_mask = np.load('../mask/north/BINMASKG.npy')
    smooth_b(df=df, bl_std_curl=bl_std_curl, bl_curl=bl_curl, m_list=cmb_list, bin_mask=bin_mask, apo_mask=None, lmax=lmax, nside=nside, save_path='./eblc_data/smsim')


