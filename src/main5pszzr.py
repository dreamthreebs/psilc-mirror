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

def nilc_pp(lmax_nilc, nside_nilc, Rtol, path_apo_mask, path_bin_mask, path_sim, path_fg, path_noise, needlet_config, save_path_nilc, number):
    from eblcilc.nilc import NILC
    apo_mask = np.load(path_apo_mask)
    bin_mask = np.load(path_bin_mask)
    sim = np.load(path_sim) * apo_mask
    fg = np.load(path_fg) * apo_mask
    noise = np.load(path_noise) * apo_mask

    if not os.path.exists(save_path_nilc):
        os.makedirs(save_path_nilc)
    weights_name = os.path.join(save_path_nilc, f'weight{number}.npz')

    obj = NILC(needlet_config=needlet_config, Sm_alms=None, weights_name=weights_name, Sm_maps=sim, lmax=lmax_nilc, nside=nside_nilc, Rtol=Rtol)

    ilc_res = obj.run_nilc()

    np.save(os.path.join(save_path_nilc, f'nilc_map{number}.npy'), ilc_res)

    obj = NILC(needlet_config=needlet_config, Sm_alms=None, weights_config=weights_name, Sm_maps=fg, lmax=lmax_nilc, nside=nside_nilc)
    fg_res = obj.run_nilc()

    np.save(os.path.join(save_path_nilc, f'nilc_fgres_map{number}.npy'), fg_res)

    obj = NILC(needlet_config=needlet_config, Sm_alms=None, weights_config=weights_name, Sm_maps=noise, lmax=lmax_nilc, nside=nside_nilc)
    noise_res = obj.run_nilc()

    np.save(os.path.join(save_path_nilc, f'nilc_noise_res_map{number}.npy'), noise_res)

def hilc_pp(lmax_hilc, nside_hilc, path_apo_mask, path_bin_mask, path_sim, path_fg, path_noise, save_path_hilc):
    from eblcilc.hilc import harmonic_ilc

    apo_mask = np.load(path_apo_mask)
    bin_mask = np.load(path_bin_mask)
    sim = np.load(path_sim) * apo_mask
    fg = np.load(path_fg) * apo_mask
    noise = np.load(path_noise) * apo_mask
    print(f'sim.shape = {sim.shape}')

    wl, ilc_alm = harmonic_ilc(sim, lmax=lmax_hilc, nside=nside_hilc)
    ilc_map = hp.alm2map(ilc_alm, nside=nside_hilc)

    _, fg_res_alm = harmonic_ilc(fg, lmax=lmax_hilc, nside=nside_hilc, wl=wl)
    fgres_map = hp.alm2map(fg_res_alm, nside=nside_hilc)

    _, noise_res_alm = harmonic_ilc(noise, lmax=lmax_hilc, nside=nside_hilc, wl=wl)
    noise_res_map = hp.alm2map(noise_res_alm, nside=nside_hilc)

    if not os.path.exists(save_path_hilc):
        os.makedirs(save_path_hilc)

    np.save(os.path.join(save_path_hilc, 'wl.npy'), wl)
    np.save(os.path.join(save_path_hilc, 'hilc_map.npy'), ilc_map)
    np.save(os.path.join(save_path_hilc, 'hilc_fgres_map.npy'), fgres_map)
    np.save(os.path.join(save_path_hilc, 'hilc_noise_res_map.npy'), noise_res_map)

def pilc_pp(lmax_pilc, nside_pilc, path_apo_mask, path_bin_mask, path_sim, path_fg, path_noise, save_path_pilc):
    from eblcilc.pilc import pixel_ilc

    apo_mask = np.load(path_apo_mask)
    bin_mask = np.load(path_bin_mask)

    sim = np.load(path_sim) * apo_mask
    fg = np.load(path_fg) * apo_mask
    noise = np.load(path_noise) * apo_mask

    weight, ilc_res = pixel_ilc(sim)
    fgres = weight @ fg
    noiseres = weight @ noise

    if not os.path.exists(save_path_pilc):
        os.makedirs(save_path_pilc)

    np.save(os.path.join(save_path_pilc, 'w.npy'), weight)
    np.save(os.path.join(save_path_pilc, 'pilc_map.npy'), ilc_res)
    np.save(os.path.join(save_path_pilc, 'pilc_fgres_map.npy'), fgres)
    np.save(os.path.join(save_path_pilc, 'pilc_noise_res_map.npy'), noiseres)



if __name__ == '__main__':

    # eblc_at_diff_freq(lmax_eblc=350, nside_eblc=512, glob_path='./sim/NSIDE512BAND5/PS/CMB', save_path_eblc='../newdata/band5ps350/eblc/cmb', method='zzr', path_bin_mask_eblc='./mask/north/BINMASKG.npy')
    # eblc_at_diff_freq(lmax_eblc=350, nside_eblc=512, glob_path='./sim/NSIDE512BAND5/PS/SIM', save_path_eblc='../newdata/band5ps350/eblc/sim', method='zzr', path_bin_mask_eblc='./mask/north/BINMASKG.npy')
    # eblc_at_diff_freq(lmax_eblc=350, nside_eblc=512, glob_path='./sim/NSIDE512BAND5/PS/FG', save_path_eblc='../newdata/band5ps350/eblc/fg', method='zzr', path_bin_mask_eblc='./mask/north/BINMASKG.npy')
    # eblc_at_diff_freq(lmax_eblc=350, nside_eblc=512, glob_path='./sim/NSIDE512BAND5/PS/NOISE', save_path_eblc='../newdata/band5ps350/eblc/noise', method='zzr', path_bin_mask_eblc='./mask/north/BINMASKG.npy')

    # smooth_eblc_result(lmax_sm=350, nside_sm=512, beam_base=63, path_df='../FGSim/FreqBand5', glob_path='../newdata/band5ps350/eblc/cmb', save_path_sm='../newdata/band5ps350/smcmb', path_bin_mask_sm='./mask/north/BINMASKG.npy')
    # smooth_eblc_result(lmax_sm=350, nside_sm=512, beam_base=63, path_df='../FGSim/FreqBand5', glob_path='../newdata/band5ps350/eblc/sim', save_path_sm='../newdata/band5ps350/smsim', path_bin_mask_sm='./mask/north/BINMASKG.npy')
    # smooth_eblc_result(lmax_sm=350, nside_sm=512, beam_base=63, path_df='../FGSim/FreqBand5', glob_path='../newdata/band5ps350/eblc/fg', save_path_sm='../newdata/band5ps350/smfg', path_bin_mask_sm='./mask/north/BINMASKG.npy')
    # smooth_eblc_result(lmax_sm=350, nside_sm=512, beam_base=63, path_df='../FGSim/FreqBand5', glob_path='../newdata/band5ps350/eblc/noise', save_path_sm='../newdata/band5ps350/smnoise', path_bin_mask_sm='./mask/north/BINMASKG.npy')

    nilc_pp(lmax_nilc=350, nside_nilc=512, Rtol=1/1000, path_apo_mask='./mask/north/APOMASKC1_5.npy', path_bin_mask='./mask/north/BINMASKG.npy', path_sim=f'../newdata/band5ps350/smsim/data.npy', path_fg=f'../newdata/band5ps350/smfg/data.npy', path_noise=f'../newdata/band5ps350/smnoise/data.npy', needlet_config=f'./eblcilc/needlets/needlet3.csv', save_path_nilc=f'../newdata/band5ps350/simnilc', number=0)
    hilc_pp(lmax_hilc=350, nside_hilc=512, path_apo_mask='./mask/north/APOMASKC1_5.npy', path_bin_mask='./mask/north/BINMASKG.npy', path_sim=f'../newdata/band5ps350/smsim/data.npy', path_fg='../newdata/band5ps350/smfg/data.npy', path_noise=f'../newdata/band5ps350/smnoise/data.npy', save_path_hilc='../newdata/band5ps350/simhilc')
    pilc_pp(lmax_pilc=350, nside_pilc=512, path_apo_mask='./mask/north/APOMASKC1_5.npy', path_bin_mask='./mask/north/BINMASKG.npy', path_sim=f'../newdata/band5ps350/smsim/data.npy', path_fg='../newdata/band5ps350/smfg/data.npy', path_noise=f'../newdata/band5ps350/smnoise/data.npy', save_path_pilc='../newdata/band5ps350/simpilc')





