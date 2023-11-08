import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
from pathlib import Path

from eblc.eblc_base import EBLeakageCorrection
from eblc.eblc import EBLeakageCorrectionPipeline
from eblc.smooth_b import smooth_b, bl_curl_creater

def eblc_at_diff_freq(lmax_eblc, nside_eblc, glob_path, save_path_eblc, method, path_bin_mask_eblc):
    bin_mask_eblc = np.load(path_bin_mask_eblc)
    sim_all = glob.glob(os.path.join(glob_path, '*.npy'))
    sorted_sim = sorted(sim_all, key=lambda x: int(Path(x).stem))
    print(f'{sorted_sim=}')

    eblc_obj = EBLeakageCorrectionPipeline(method=method, m_list=sorted_sim, lmax=lmax_eblc, nside=nside_eblc, bin_mask=bin_mask_eblc, apo_mask=None, save_path=save_path_eblc)
    eblc_obj.class_main()

def smooth_eblc_result(lmax_sm, nside_sm, beam_base, path_df, glob_path, save_path_sm, path_bin_mask_sm):
    df = pd.read_csv(path_df)
    bl_std_curl = hp.gauss_beam(fwhm=np.deg2rad(beam_base/60), lmax=lmax_sm, pol=True)[:,2]
    bl_curl = bl_curl_creater(df=df, lmax=lmax_sm)
    bin_mask_sm = np.load(path_bin_mask_sm)

    sim_list = sorted(glob.glob(os.path.join(glob_path,'*.npy')), key=lambda x: int(Path(x).stem))
    smooth_b(df=df, bl_std_curl=bl_std_curl, bl_curl=bl_curl, m_list=sim_list, bin_mask=bin_mask_sm, apo_mask=None, lmax=lmax_sm, nside=nside_sm, save_path=save_path_sm)

def nilc_noisepp(lmax_nilc, nside_nilc, Rtol, path_apo_mask, path_bin_mask, path_noise, needlet_config, load_path_weight, save_path_nilc, number):
    from eblcilc.nilc import NILC
    apo_mask = np.load(path_apo_mask)
    bin_mask = np.load(path_bin_mask)
    noise = np.load(path_noise) * apo_mask

    obj = NILC(needlet_config=needlet_config, Sm_alms=None, weights_config=load_path_weight, Sm_maps=noise, lmax=lmax_nilc, nside=nside_nilc)
    noise_res = obj.run_nilc()

    if not os.path.exists(save_path_nilc):
        os.makedirs(save_path_nilc)

    np.save(os.path.join(save_path_nilc, f'nilc_noise_res_map{number}.npy'), noise_res)

def hilc_noisepp(lmax_hilc, nside_hilc, path_apo_mask, path_bin_mask, path_noise, load_path_wl, save_path_hilc, number):
    from eblcilc.hilc import harmonic_ilc

    apo_mask = np.load(path_apo_mask)
    bin_mask = np.load(path_bin_mask)
    noise = np.load(path_noise) * apo_mask

    wl = np.load(load_path_wl)
    _, noise_res_alm = harmonic_ilc(noise, lmax=lmax_hilc, nside=nside_hilc, wl=wl)
    noise_res_map = hp.alm2map(noise_res_alm, nside=nside_hilc)

    if not os.path.exists(save_path_hilc):
        os.makedirs(save_path_hilc)

    np.save(os.path.join(save_path_hilc, f'hilc_noise_res_map{number}.npy'), noise_res_map)

def pilc_noisepp(lmax_pilc, nside_pilc, path_apo_mask, path_bin_mask, path_noise, load_path_w, save_path_pilc, number):
    from eblcilc.pilc import pixel_ilc

    apo_mask = np.load(path_apo_mask)
    bin_mask = np.load(path_bin_mask)

    noise = np.load(path_noise) * apo_mask

    weight = np.load(load_path_w)
    noiseres = weight @ noise

    if not os.path.exists(save_path_pilc):
        os.makedirs(save_path_pilc)

    np.save(os.path.join(save_path_pilc, f'pilc_noise_res_map{number}.npy'), noiseres)



if __name__ == '__main__':

    n_sim_start = 10
    n_sim_end = 15

    for i in range(n_sim_start, n_sim_end):
        eblc_at_diff_freq(lmax_eblc=350, nside_eblc=512, glob_path=f'./sim/NSIDE512BAND5/NOISESIM/{i}', save_path_eblc=f'../data/noapo/eblc/NOISESIM/{i}', method='cutqufitqu', path_bin_mask_eblc='./mask/north/BINMASKG.npy')

    for i in range(n_sim_start, n_sim_end):
        smooth_eblc_result(lmax_sm=350, nside_sm=512, beam_base=63, path_df='../FGSim/FreqBand5', glob_path=f'../data/noapo/eblc/NOISESIM/{i}', save_path_sm=f'../data/band5ps/SMNOISESIM/{i}', path_bin_mask_sm='./mask/north/BINMASKG.npy')

    for i in range(n_sim_start, n_sim_end):
        nilc_noisepp(lmax_nilc=350, nside_nilc=512, Rtol=1/1000, path_apo_mask='./mask/north/APOMASKC1_10.npy', path_bin_mask='./mask/north/BINMASKG.npy', path_noise=f'../data/band5ps/SMNOISESIM/{i}/data.npy', needlet_config=f'./eblcilc/needlets/needlet2.csv', load_path_weight='../data/band5ps/simnilc/weight0.npz', save_path_nilc=f'../data/band5ps/NOISENILC', number=i)

    for i in range(n_sim_start, n_sim_end):
        print(f'{i = }')
        hilc_noisepp(lmax_hilc=350, nside_hilc=512, path_apo_mask='./mask/north/APOMASKC1_10.npy', path_bin_mask='./mask/north/BINMASKG.npy', path_noise=f'../data/band5ps/SMNOISESIM/{i}/data.npy', load_path_wl='../data/band5ps/simhilc/wl.npy', save_path_hilc='../data/band5ps/NOISEHILC', number=i)

    for i in range(n_sim_start, n_sim_end):
        print(f'{i = }')
        pilc_noisepp(lmax_pilc=350, nside_pilc=512, path_apo_mask='./mask/north/APOMASKC1_10.npy', path_bin_mask='./mask/north/BINMASKG.npy', path_noise=f'../data/band5ps/SMNOISESIM/{i}/data.npy', load_path_w='../data/band5ps/simpilc/w.npy', save_path_pilc='../data/band5ps/NOISEPILC', number=i)





