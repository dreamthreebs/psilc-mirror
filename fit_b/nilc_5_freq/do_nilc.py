import numpy as np
import healpy as hp
import pandas as pd
import time
import matplotlib.pyplot as plt
import pymaster as nmt
from pathlib import Path
from nilc import NILC
from eblc_base_slope import EBLeakageCorrection

freq_list = [30, 95, 155, 215, 270]
beam_list = [67, 30, 17, 11, 9]
beam_base = 17 #arcmin
lmax = 1500
nside = 2048
rlz_idx = 0

def calc_lmax():
    for freq, beam in zip(freq_list, beam_list):
        lmax = int(2 * np.pi / np.deg2rad(beam) * 60) + 1
        print(f'{freq=}, {beam=}, {lmax=}')

def collect_diff_freq_maps():
    map_list = []
    for freq in freq_list:
        m = np.load(f'../{freq}GHz/fit_res/sm/noise/n/0.npy')
        # m = np.load(f'../30GHz/fit_res/sm/noise/n/0.npy')
        map_list.append(m)
    map_arr = np.asarray(map_list)
    print(f'{map_arr.shape=}')
    np.save('./test/n.npy', map_arr)

def try_nilc():
    sim = np.load('./test/pcfn.npy')
    noise = np.load('./test/n.npy')
    mask = np.load(f'../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5APO_3APO_5.npy')

    # do nilc and save the weights in map
    time0 = time.time()
    obj_nilc = NILC(bandinfo='./band_info.csv', needlet_config='./needlets/0.csv', weights_name='./test/weight/pcfn.npz', Sm_maps=sim, mask=mask, lmax=lmax, nside=nside, n_iter=3, weight_in_alm=False)
    clean_map = obj_nilc.run_nilc()
    np.save('./test/cln_pcfn.npy', clean_map)
    print(f'{time.time()-time0=}')

    # obj_noise = NILC(bandinfo='./band_info.csv', needlet_config='./needlets/0.csv', weights_config='./test/weight/pcfn.npz', Sm_maps=noise, mask=mask, lmax=lmax, nside=nside, n_iter=3, weight_in_alm=False)
    # cln_n = obj_noise.run_nilc()
    # np.save('./test/cln_n.npy', cln_n)

def gen_fiducial_cmb():
    cmb_seed = np.load('../seeds_cmb_2k.npy')
    cls = np.load('../../src/cmbsim/cmbdata/cmbcl_8k.npy')
    np.random.seed(seed=cmb_seed[0])
    cmb_iqu = hp.synfast(cls.T, nside=nside, fwhm=np.deg2rad(beam_base)/60, new=True, lmax=3*nside-1)
    cl_cmb_b = hp.anafast(cmb_iqu, lmax=lmax)[2]
    return cl_cmb_b

def check_try_nilc():
    mask = np.load(f'../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5APO_3APO_5.npy')
    fsky = np.sum(mask) / np.size(mask)
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam_base)/60, lmax=lmax, pol=True)[:,2]
    df_ps = pd.read_csv(f'../95GHz/mask/95_after_filter.csv')
    cln_pcfn = np.load(f'./test/cln_pcfn.npy')
    cln_cfn = np.load(f'./test/cln_cfn.npy')
    cln_cf = np.load(f'./test/cln_cf.npy')
    cln_n = np.load(f'./test/cln_n.npy')
    hp.orthview(cln_pcfn, rot=[100,50,0], min=-0.7, max=0.7)
    hp.orthview(cln_cfn, rot=[100,50,0], min=-0.7, max=0.7)
    hp.orthview(cln_cf, rot=[100,50,0], min=-0.7, max=0.7)
    hp.orthview(cln_cfn-cln_pcfn, rot=[100,50,0])
    hp.orthview(cln_n, rot=[100,50,0])
    plt.show()

    # for flux_idx in np.arange(len(df_ps)):
    #     lon = np.rad2deg(df_ps.at[flux_idx, 'lon'])
    #     lat = np.rad2deg(df_ps.at[flux_idx, 'lat'])
    #     hp.gnomview(cln_pcfn, rot=[lon, lat, 0], title='pcfn')
    #     hp.gnomview(cln_cfn, rot=[lon, lat, 0], title='cfn')
    #     hp.gnomview(cln_cf, rot=[lon, lat, 0], title='cf')
    #     plt.show()

    cl_pcfn = hp.anafast(cln_pcfn, lmax=lmax)
    cl_n = hp.anafast(cln_n, lmax=lmax)
    # cl_cfn = hp.anafast(cln_cfn, lmax=lmax)
    cl_cf = hp.anafast(cln_cf, lmax=lmax)
    cl_fid = gen_fiducial_cmb()
    l = np.arange(np.size(cl_pcfn))
    plt.loglog(l, l*(l+1)*(cl_pcfn-cl_n)/bl**2/fsky, label='debias pcfn')
    plt.loglog(l, l*(l+1)*cl_n/bl**2/fsky, label='noise')
    # plt.loglog(l, l*(l+1)*cl_cfn/bl**2/fsky, label='cfn')
    plt.loglog(l, l*(l+1)*cl_cf/bl**2/fsky, label='cf')
    plt.loglog(l, l*(l+1)*cl_fid/bl**2, label='fiducial')
    plt.legend()
    plt.show()

def do_nilc():

    mask = np.load(f'../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5APO_3APO_5.npy')
    # m = np.load(f'../30GHz/fit_res/sm/mean/cf/n_/{rlz_idx}.npy')

    def _calc(sim_mode, method):
        pcfn = np.asarray([np.load(f'../{freq}GHz/fit_res/sm/{sim_mode}/{method}/{rlz_idx}.npy') for freq in freq_list])
        if method == 'pcfn' or method == 'cfn':
            n = np.asarray([np.load(f'../{freq}GHz/fit_res/sm/noise/n/{rlz_idx}.npy') for freq in freq_list])
        elif method == 'inp':
            n = np.asarray([np.load(f'../{freq}GHz/fit_res/sm/noise/n_inp/{rlz_idx}.npy') for freq in freq_list])
        elif method == 'rmv':
            n = np.asarray([np.load(f'../{freq}GHz/fit_res/sm/noise/n_rmv/{rlz_idx}.npy') for freq in freq_list])

        # do nilc and save the weights in map
        time0 = time.time()
        obj_nilc = NILC(bandinfo='./band_info.csv', needlet_config='./needlets/0.csv', weights_name=f'./weight/{sim_mode}/{method}/{rlz_idx}.npz', Sm_maps=pcfn, mask=mask, lmax=lmax, nside=nside, n_iter=3, weight_in_alm=False)
        cln_pcfn = obj_nilc.run_nilc()
        Path(f'./data/{sim_mode}/{method}').mkdir(exist_ok=True, parents=True)
        np.save(f'./data/{sim_mode}/{method}/{rlz_idx}.npy', cln_pcfn)
        print(f'{time.time()-time0=}')

        obj_noise = NILC(bandinfo='./band_info.csv', needlet_config='./needlets/0.csv', weights_config=f'./weight/{sim_mode}/{method}/{rlz_idx}.npz', Sm_maps=n, mask=mask, lmax=lmax, nside=nside, n_iter=3, weight_in_alm=False)
        cln_n = obj_noise.run_nilc()
        Path(f'./data/{sim_mode}/n_{method}').mkdir(exist_ok=True, parents=True)
        np.save(f'./data/{sim_mode}/n_{method}/{rlz_idx}.npy', cln_n)

    print(f'MAN, do pcfn nilc!')
    _calc(sim_mode='mean', method='pcfn')
    _calc(sim_mode='std', method='pcfn')
    print(f'MAN, do cfn nilc!')
    _calc(sim_mode='mean', method='cfn')
    _calc(sim_mode='std', method='cfn')
    print(f'MAN, do inp nilc!')
    _calc(sim_mode='mean', method='inp')
    _calc(sim_mode='std', method='inp')
    print(f'MAN, do rmv nilc!')
    _calc(sim_mode='mean', method='rmv')
    _calc(sim_mode='std', method='rmv')


def check_do_pcfn():
    mask = np.load(f'../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5APO_3APO_5.npy')
    fsky = np.sum(mask) / np.size(mask)
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam_base)/60, lmax=lmax, pol=True)[:,2]
    df_ps = pd.read_csv(f'../95GHz/mask/95_after_filter.csv')
    cln_pcfn = np.load(f'./data/mean/pcfn/0.npy')
    cln_cfn = np.load(f'./data/mean/cfn/0.npy')
    cln_rmv = np.load(f'./data/mean/rmv/0.npy')
    cln_inp = np.load(f'./data/mean/inp/0.npy')
    # cln_n = np.load(f'./data/mean/n/0.npy')
    hp.orthview(cln_pcfn, rot=[100,50,0], title='pcfn')
    hp.orthview(cln_cfn, rot=[100,50,0], title='cfn')
    hp.orthview(cln_rmv, rot=[100,50,0], title='rmv')
    hp.orthview(cln_inp, rot=[100,50,0], title='inp')
    # hp.orthview(cln_n, rot=[100,50,0])
    plt.show()

    for flux_idx in np.arange(len(df_ps)):
        lon = np.rad2deg(df_ps.at[flux_idx, 'lon'])
        lat = np.rad2deg(df_ps.at[flux_idx, 'lat'])
        hp.gnomview(cln_pcfn, rot=[lon, lat, 0], title='pcfn')
        hp.gnomview(cln_cfn, rot=[lon, lat, 0], title='cfn')
        hp.gnomview(cln_rmv, rot=[lon, lat, 0], title='rmv')
        hp.gnomview(cln_inp, rot=[lon, lat, 0], title='inp')
        # hp.gnomview(cln_cf, rot=[lon, lat, 0], title='cf')
        plt.show()

    # cl_pcfn = hp.anafast(cln_pcfn, lmax=lmax)
    # cl_n = hp.anafast(cln_n, lmax=lmax)
    # cl_fid = gen_fiducial_cmb()
    # l = np.arange(np.size(cl_pcfn))
    # plt.loglog(l, l*(l+1)*(cl_pcfn-cl_n)/bl**2/fsky, label='debias pcfn')
    # plt.loglog(l, l*(l+1)*cl_n/bl**2/fsky, label='noise')
    # plt.loglog(l, l*(l+1)*cl_fid/bl**2, label='fiducial')
    # plt.legend()
    # plt.show()

def calc_dl_from_scalar_map(scalar_map, apo_mask, bin_dl, masked_on_input):
    scalar_field = nmt.NmtField(apo_mask, [scalar_map], masked_on_input=masked_on_input, lmax=lmax, lmax_mask=lmax)
    dl = nmt.compute_full_master(scalar_field, scalar_field, bin_dl)
    return dl[0]

def generate_bins(l_min_start=30, delta_l_min=30, l_max=1500, fold=0.3):
    bins_edges = []
    l_min = l_min_start  # starting l_min

    while l_min < l_max:
        delta_l = max(delta_l_min, int(fold * l_min))
        l_next = l_min + delta_l
        bins_edges.append(l_min)
        l_min = l_next

    # Adding l_max to ensure the last bin goes up to l_max
    bins_edges.append(l_max)
    return bins_edges[:-1], bins_edges[1:]

def gen_fiducial_cmb():
    l_min_edges, l_max_edges = generate_bins(l_min_start=30, delta_l_min=30, l_max=lmax+1, fold=0.2)
    # delta_ell = 30
    # bin_dl = nmt.NmtBin.from_nside_linear(nside, nlb=delta_ell, is_Dell=True)
    # bin_dl = nmt.NmtBin.from_lmax_linear(lmax=lmax, nlb=30, is_Dell=True)
    bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    ell_arr = bin_dl.get_effective_ells()

    cmb_seed = np.load(f'../seeds_cmb_2k.npy')
    cls = np.load(f'../../src/cmbsim/cmbdata/cmbcl_8k.npy')
    np.random.seed(seed=cmb_seed[rlz_idx])
    cmb_iqu = hp.synfast(cls.T, nside=nside, new=True, lmax=3*nside-1)
    mask = np.load('../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5.npy')
    obj_eblc = EBLeakageCorrection(m=cmb_iqu, lmax=lmax, nside=nside, mask=mask, post_mask=mask)
    _, _, cln_cmb = obj_eblc.run_eblc()

    hp.orthview(cln_cmb, rot=[100,50,0])
    plt.show()


    # mask_cl = np.load(f'../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5APO_3APO_5APO_3.npy')
    # dl_cmb = calc_dl_from_scalar_map(scalar_map=cln_cmb, apo_mask=mask_cl, bin_dl=bin_dl, masked_on_input=False)
    # path_fid_cmb = Path(f'./dl_res/fid_cmb')
    # path_fid_cmb.mkdir(exist_ok=True, parents=True)
    # np.save(path_fid_cmb / Path(f'{rlz_idx}.npy'), dl_cmb)

if __name__ == "__main__":
    # calc_lmax()
    # collect_diff_freq_maps()
    try_nilc()
    # check_try_nilc()
    # do_nilc()
    # check_do_pcfn()
    # gen_fiducial_cmb()

    pass




