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

def nilc_pp(lmax_nilc, nside_nilc, Rtol, path_apo_mask, path_bin_mask, path_sim, path_fg, needlet_config, save_path_nilc, number):
    from eblcilc.nilc import NILC
    apo_mask = np.load(path_apo_mask)
    bin_mask = np.load(path_bin_mask)
    sim = np.load(path_sim) * apo_mask
    fg = np.load(path_fg) * apo_mask
    # fgnoise = np.load(f'../sim/simdata/{cl_type}/fgnoise.npy')
    # cmb = np.load(f'../sim/simdata/{cl_type}/cmb.npy')
    # noise = np.load(f'../smooth/FULL_PATCH/PS_northLOWNOI/NOISE/{cl_type}/data.npy')

    if not os.path.exists(save_path_nilc):
        os.makedirs(save_path_nilc)
    weights_name = os.path.join(save_path_nilc, f'weight{number}.npz')

    obj = NILC(needlet_config=needlet_config, Sm_alms=None, weights_name=weights_name, Sm_maps=sim, lmax=lmax_nilc, nside=nside_nilc, Rtol=Rtol)
    # obj = NILC(needlet_config='./needlets/needlet.csv', Sm_alms=None, weights_config=f'./nilcdata/weightexact.npz', Sm_maps=fg, lmax=lmax, nside=nside)
    
    ilc_res = obj.run_nilc()
    
    np.save(os.path.join(save_path_nilc, f'nilc_map{number}.npy'), ilc_res)
    
    obj = NILC(needlet_config=needlet_config, Sm_alms=None, weights_config=weights_name, Sm_maps=fg, lmax=lmax_nilc, nside=nside_nilc)
    fg_res = obj.run_nilc()

    np.save(os.path.join(save_path_nilc, f'nilc_fgres_map{number}.npy'), fg_res)

    # # noise = np.load(f'../smooth/FULL_SKY/SM_NOISE/{cl_type}/noise.npy')
    # obj = NILC(needlet_config=f'./needlets/needlet{needlet_type}.csv', Sm_alms=None, weights_config=f'./FULL_{cl_type}/weight{number}.npz', Sm_maps=noise, lmax=lmax, nside=nside)
    # noise_res = obj.run_nilc()
    # noiseres_cl = hp.anafast(noise_res,lmax=lmax)
    # np.save(f'./FULL_{cl_type}/nilc_noise_cl{number}.npy', noiseres_cl)
    
    
    # for i in range(15,30):
    #     print(f'{i}')
    #     noise = np.load(f'../smooth/FULL_SKY/SM_NOISE/{i}/{cl_type}/noise.npy')
    #     obj = NILC(needlet_config=f'./needlets/needlet{needlet_type}.csv', Sm_alms=None, weights_config=f'./FULL_{cl_type}/weight{number}.npz', Sm_maps=noise, lmax=lmax, nside=nside)
    #     noise_res = obj.run_nilc()
    #     noiseres_cl = hp.anafast(noise_res,lmax=lmax)
    #     np.save(f'./FULL_B/NOISE/nilc_noise_cl{number}{i}.npy', noiseres_cl)

    # obj = NILC(needlet_config=f'./needlets/needlet{needlet_type}.csv', Sm_alms=None, weights_config=f'./{cl_type}/weight{number}.npz', Sm_maps=fgnoise, lmax=lmax, nside=nside)
    # d = obj.run_nilc()
    # obj = NILC(needlet_config=f'./needlets/needlet{needlet_type}.csv', Sm_alms=None, weights_config=f'./{cl_type}/weight{number}.npz', Sm_maps=cmb, lmax=lmax, nside=nside)
    # s = obj.run_nilc()
    
    # cl_sd = 2 * hp.anafast(s,d,lmax=lmax)
    # np.save(f'./{cl_type}/nilc_clsd{number}.npy', cl_sd)
    
def hilc_pp(lmax_hilc, nside_hilc, path_apo_mask, path_bin_mask, path_sim, path_fg, save_path_hilc):
    from eblcilc.hilc import harmonic_ilc
    
    apo_mask = np.load(path_apo_mask)
    bin_mask = np.load(path_bin_mask)
    sim = np.load(path_sim) * apo_mask
    fg = np.load(path_fg) * apo_mask
    # noise = np.load(f'../smooth/FULL_PATCH/noPS_northNOI/NOISE/B/data.npy')
    # bl = np.load('../smooth/BL/bl_std_curl.npy')
    
    print(f'sim.shape = {sim.shape}')

    wl, ilc_alm = harmonic_ilc(sim, lmax=lmax_hilc, nside=nside_hilc)
    ilc_map = hp.alm2map(ilc_alm, nside=nside_hilc)
    
    _, fg_res_alm = harmonic_ilc(fg, lmax=lmax_hilc, nside=nside_hilc, wl=wl)
    fgres_map = hp.alm2map(fg_res_alm, nside=nside_hilc)
    
    # _, noise_res_alm = harmonic_ilc(noise, wl=wl)

    if not os.path.exists(save_path_hilc):
        os.makedirs(save_path_hilc)
    
    np.save(os.path.join(save_path_hilc, 'wl.npy'), wl)
    np.save(os.path.join(save_path_hilc, 'hilc_map.npy'), ilc_map)
    np.save(os.path.join(save_path_hilc, 'hilc_fgres_map.npy'), fgres_map)

def pilc_pp(lmax_pilc, nside_pilc, path_apo_mask, path_bin_mask, path_sim, path_fg, save_path_pilc):
    from eblcilc.pilc import pixel_ilc

    apo_mask = np.load(path_apo_mask)
    bin_mask = np.load(path_bin_mask)

    sim = np.load(path_sim) * apo_mask
    fg = np.load(path_fg) * apo_mask

    weight, ilc_res = pixel_ilc(sim)
    fgres = weight @ fg
    # noiseres = weight @ noise

    if not os.path.exists(save_path_pilc):
        os.makedirs(save_path_pilc)

    np.save(os.path.join(save_path_pilc, 'w.npy'), weight)
    np.save(os.path.join(save_path_pilc, 'pilc_map.npy'), ilc_res)
    np.save(os.path.join(save_path_pilc, 'pilc_fgres_map.npy'), fgres)

    # n_sim = 50
    # noise_cl_sum = 0
    # for i in range(n_sim):
    #     print(f'loop:{i}')
    #     noise = np.load(f'../smooth/FULL_PATCH/noPS_northNOI/NOISESIM/{i}/{cl_type}/data.npy')
    #     noiseres = weight @ noise
    #     noise_res_cl = hp.anafast(noiseres, lmax=lmax)
    #     noise_cl_sum = noise_cl_sum + noise_res_cl
    
    # noise_res_avg = noise_cl_sum / n_sim
    # np.save('./pilcres/pilc_noise_avg2',noise_res_avg)


if __name__ == '__main__':

    # eblc_at_diff_freq(lmax_eblc=500, nside_eblc=512, glob_path='./sim/NSIDE512/noPS/CMB', save_path_eblc='../data/sim300/eblc/cmb', method='cutqufitqu', path_bin_mask_eblc='./mask/north/BINMASKG.npy')
    # eblc_at_diff_freq(lmax_eblc=500, nside_eblc=512, glob_path='./sim/NSIDE512/noPS/SIM', save_path_eblc='../data/sim300/eblc/sim', method='cutqufitqu', path_bin_mask_eblc='./mask/north/BINMASKG.npy')
    # eblc_at_diff_freq(lmax_eblc=500, nside_eblc=512, glob_path='./sim/NSIDE512/noPS/FG', save_path_eblc='../data/sim300/eblc/fg', method='cutqufitqu', path_bin_mask_eblc='./mask/north/BINMASKG.npy')

    # smooth_eblc_result(lmax_sm=350, nside_sm=512, beam_base=9, path_df='../FGSim/FreqBand', glob_path='../data/sim300/eblc/sim', save_path_sm='../data/sim300/smsim', path_bin_mask_sm='./mask/north/BINMASKG.npy')
    # smooth_eblc_result(lmax_sm=350, nside_sm=512, beam_base=9, path_df='../FGSim/FreqBand', glob_path='../data/sim300/eblc/fg', save_path_sm='../data/sim300/smfg', path_bin_mask_sm='./mask/north/BINMASKG.npy')

    # nilc_pp(lmax_nilc=350, nside_nilc=512, Rtol=1/100, path_apo_mask='./mask/north/APOMASKC1_10.npy', path_bin_mask='./mask/north/BINMASKG.npy', path_sim=f'../data/sim300/smsim/data.npy', path_fg=f'../data/sim300/smfg/data.npy', needlet_config=f'./eblcilc/needlets/needlet2.csv', save_path_nilc=f'../data/sim300/sim300nilc', number=0)

    hilc_pp(lmax_hilc=350, nside_hilc=512, path_apo_mask='./mask/north/APOMASKC1_10.npy', path_bin_mask='./mask/north/BINMASKG.npy', path_sim=f'../data/sim300/smsim/data.npy', path_fg='../data/sim300/smfg/data.npy', save_path_hilc='../data/sim300/sim300hilc')
    # pilc_pp(lmax_pilc=350, nside_pilc=512, path_apo_mask='./mask/north/APOMASKC1_10.npy', path_bin_mask='./mask/north/BINMASKG.npy', path_sim=f'../data/sim300/smsim/data.npy', path_fg='../data/sim300/smfg/data.npy', save_path_pilc='../data/sim300/sim300pilc')





