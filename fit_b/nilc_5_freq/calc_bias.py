import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt
import pymaster as nmt
from pathlib import Path
from nilc import NILC
from eblc_base_slope import EBLeakageCorrection
rlz_idx = 0
freq_list = [30, 95, 155, 215, 270]
beam_list = [67, 30, 17, 11, 9]
lmax_list = [500, 1300, 1800, 2500, 3500]
beam_base = 17 #arcmin
lmax = 1500
nside = 2048

cmb_seed = np.load('../seeds_cmb_2k.npy')
noise_seed = np.load(f'../seeds_noise_2k.npy')
apo_mask = np.load('../../psfit/fitv4/fit_res/2048/ps_mask/new_mask/apo_C1_3_apo_3_apo_3.npy')
bin_mask = np.load('../../psfit/fitv4/fit_res/2048/ps_mask/new_mask/BIN_C1_3_C1_3.npy')
ilc_mask = np.load('../../psfit/fitv4/fit_res/2048/ps_mask/new_mask/apo_C1_3_apo_3.npy')

def calc_lmax():
    for freq, beam in zip(freq_list, beam_list):
        lmax = int(2 * np.pi / np.deg2rad(beam) * 60) + 1
        print(f'{freq=}, {beam=}, {lmax=}')

def calc_dl_from_scalar_map(scalar_map, apo_mask, bin_dl, masked_on_input):
    scalar_field = nmt.NmtField(apo_mask, [scalar_map], masked_on_input=masked_on_input, lmax=lmax, lmax_mask=lmax)
    dl = nmt.compute_full_master(scalar_field, scalar_field, bin_dl)
    return dl[0]

def calc_dl_from_scalar_map_bl(scalar_map, apo_mask, bl, bin_dl, masked_on_input):
    scalar_field = nmt.NmtField(apo_mask, [scalar_map], beam=bl, masked_on_input=masked_on_input, lmax=lmax, lmax_mask=lmax)
    dl = nmt.compute_full_master(scalar_field, scalar_field, bin_dl)
    return dl[0]

def generate_bins(l_min_start=30, delta_l_min=30, l_max=1500, fold=0.3, l_threshold=None):
    bins_edges = []
    l_min = l_min_start  # starting l_min

    # Fixed binning until l_threshold if provided
    if l_threshold is not None:
        while l_min < l_threshold:
            l_next = l_min + delta_l_min
            if l_next > l_threshold:
                break
            bins_edges.append(l_min)
            l_min = l_next

    # Transition to dynamic binning
    while l_min < l_max:
        delta_l = max(delta_l_min, int(fold * l_min))
        l_next = l_min + delta_l
        bins_edges.append(l_min)
        l_min = l_next

    # Adding l_max to ensure the last bin goes up to l_max
    bins_edges.append(l_max)
    return bins_edges[:-1], bins_edges[1:]

def gen_map(freq, rlz_idx=0, mode='mean', return_noise=False):
    # mode can be mean or std
    nside = 2048

    nstd = np.load(f'../../FGSim/NSTDNORTH/2048/{freq}.npy')
    npix = hp.nside2npix(nside=2048)
    np.random.seed(seed=noise_seed[rlz_idx])
    # noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))
    noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))
    print(f"{np.std(noise[1])=}")

    if return_noise:
        return noise

    ps = np.load(f'../../fitdata/2048/PS/{freq}/ps.npy')
    fg = np.load(f'../../fitdata/2048/FG/{freq}/fg.npy')

    cls = np.load('../../src/cmbsim/cmbdata/cmbcl_8k.npy')
    if mode=='std':
        np.random.seed(seed=cmb_seed[rlz_idx])
    elif mode=='mean':
        np.random.seed(seed=cmb_seed[0])

    cmb_iqu = hp.synfast(cls.T, nside=nside, fwhm=np.deg2rad(beam)/60, new=True, lmax=3*nside-1)

    pcfn = noise + ps + cmb_iqu + fg
    cfn = noise + cmb_iqu + fg
    cf = cmb_iqu + fg
    n = noise
    return pcfn, cfn, cf, n

def gen_pcfn(freq, beam, rlz_idx=0, mode='mean', return_noise=False):
    # mode can be mean or std
    nside = 2048

    nstd = np.load(f'../../FGSim/NSTDNORTH/2048/{freq}.npy')
    npix = hp.nside2npix(nside=2048)
    np.random.seed(seed=noise_seed[rlz_idx])
    # noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))
    noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))
    print(f"{np.std(noise[1])=}")

    if return_noise:
        return noise

    ps = np.load(f'../../fitdata/2048/PS/{freq}/ps.npy')
    fg = np.load(f'../../fitdata/2048/FG/{freq}/fg.npy')

    cls = np.load('../../src/cmbsim/cmbdata/cmbcl_8k.npy')
    if mode=='std':
        np.random.seed(seed=cmb_seed[rlz_idx])
    elif mode=='mean':
        np.random.seed(seed=cmb_seed[0])

    cmb_iqu = hp.synfast(cls.T, nside=nside, fwhm=np.deg2rad(beam)/60, new=True, lmax=3*nside-1)

    pcfn = noise + ps + cmb_iqu + fg
    return pcfn

def gen_cfn(freq, beam, rlz_idx=0, mode='mean', return_noise=False):
    # mode can be mean or std
    nside = 2048

    nstd = np.load(f'../../FGSim/NSTDNORTH/2048/{freq}.npy')
    npix = hp.nside2npix(nside=2048)
    np.random.seed(seed=noise_seed[rlz_idx])
    # noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))
    noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))
    print(f"{np.std(noise[1])=}")

    if return_noise:
        return noise

    fg = np.load(f'../../fitdata/2048/FG/{freq}/fg.npy')

    cls = np.load('../../src/cmbsim/cmbdata/cmbcl_8k.npy')
    if mode=='std':
        np.random.seed(seed=cmb_seed[rlz_idx])
    elif mode=='mean':
        np.random.seed(seed=cmb_seed[0])

    cmb_iqu = hp.synfast(cls.T, nside=nside, fwhm=np.deg2rad(beam)/60, new=True, lmax=3*nside-1)

    cfn = noise + cmb_iqu + fg
    return cfn


def gen_cmb(beam, rlz_idx=0):

    cls = np.load('../../src/cmbsim/cmbdata/cmbcl_8k.npy')
    np.random.seed(seed=cmb_seed[rlz_idx])

    cmb_iqu = hp.synfast(cls.T, nside=nside, fwhm=np.deg2rad(beam)/60, new=True, lmax=3*nside-1)
    return cmb_iqu

def gen_noise(freq, rlz_idx=0):
    nside = 2048

    nstd = np.load(f'../../FGSim/NSTDNORTH/2048/{freq}.npy')
    npix = hp.nside2npix(nside=2048)
    np.random.seed(seed=noise_seed[rlz_idx])
    # noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))
    noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))
    print(f"{np.std(noise[1])=}")
    return noise



# tests on if foreground except point sources area is bigger than the foreground in point sources area
def calc_eblc_bias():
    # first check E to B leakage effects
    cmb = gen_cmb(beam=beam_base, rlz_idx=rlz_idx)
    full_b = hp.alm2map(hp.map2alm(cmb, lmax=lmax)[2], nside=nside)

    Path(f'./data3/fid_c').mkdir(exist_ok=True, parents=True)
    np.save(f'./data3/fid_c/{rlz_idx}.npy', full_b)

    # hp.orthview(full_b, rot=[100,50,0])
    # # plt.savefig('./figs/full_b.png', dpi=300)
    # plt.show()

def check_eblc_bias():
    pass

def calc_cmb_ilc_bias():
    # see cmb after ilc, compare with cmb without ilc
    pass

def check_cmb_ilc_bias():
    pass

def calc_fg_bias():
    # calc fg bias after nilc
    sim_mode = 'std'
    method = 'cfn'
    cfn = np.asarray([np.load(f'../{freq}GHz/fit_res/sm_new/{sim_mode}/{method}/{rlz_idx}.npy') for freq in freq_list])

    obj_cf = NILC(bandinfo='./band_info.csv', needlet_config='./needlets/0.csv', weights_config=f'./weight/std/pcfn/{rlz_idx}.npz', Sm_maps=cfn, mask=ilc_mask, lmax=lmax, nside=nside, n_iter=3, weight_in_alm=False)
    cln_cf = obj_cf.run_nilc()

    Path(f'./data3/cfn').mkdir(exist_ok=True, parents=True)
    np.save(f'./data3/cfn/{rlz_idx}.npy', cln_cf)

def check_fg_bias_obselete():
    cln_cf = np.load(f'./data3/cfn/{rlz_idx}.npy')
    fid_c = np.load(f'./data3/fid_c/{rlz_idx}.npy')

    # hp.orthview(cln_cf, rot=[100,50,0], half_sky=True, min=-0.7, max=0.7)
    # hp.orthview(fid_c, rot=[100,50,0], half_sky=True, min=-0.7, max=0.7)
    # hp.orthview(fid_c - cln_cf, rot=[100,50,0], half_sky=True, min=-0.7, max=0.7)
    hp.orthview((fid_c - cln_cf) * apo_mask, rot=[100,50,0], half_sky=True)
    plt.show()

def calc_fg_bias_cl_obselete():
    # get the foreground bias upon different masks
    # that might be wrong because cmb might be a little different from ilc befored maps !!!
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam_base)/60, lmax=lmax, pol=True)[:,2]
    l_min_edges, l_max_edges = generate_bins(l_min_start=42, delta_l_min=40, l_max=lmax+1, fold=0.1, l_threshold=400)
    # delta_ell = 30
    # bin_dl = nmt.NmtBin.from_nside_linear(nside, nlb=delta_ell, is_Dell=True)
    # bin_dl = nmt.NmtBin.from_lmax_linear(lmax=lmax, nlb=40, is_Dell=True)
    bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    ell_arr = bin_dl.get_effective_ells()

    diff = np.load(f'./data3/cfn/{rlz_idx}.npy') - np.load(f'./data3/fid_c/{rlz_idx}.npy')

    ps_mask = np.load(f'./ps_mask/union.npy')

    # hp.orthview(ps_mask, rot=[100,50,0], half_sky=True)
    # hp.orthview(apo_mask, rot=[100,50,0], half_sky=True)
    # plt.show()


    dl_diff_apo = calc_dl_from_scalar_map_bl(scalar_map=diff, apo_mask=apo_mask, bl=bl, bin_dl=bin_dl, masked_on_input=False)
    dl_diff_ps_mask = calc_dl_from_scalar_map_bl(scalar_map=diff, apo_mask=ps_mask, bl=bl, bin_dl=bin_dl, masked_on_input=False)

    path_dl_diff = Path(f'./dl_res5/fg_bias_cfn')
    path_dl_diff.mkdir(parents=True, exist_ok=True)

    np.save(path_dl_diff / Path(f'union_{rlz_idx}.npy'), dl_diff_ps_mask)
    np.save(path_dl_diff / Path(f'apo_{rlz_idx}.npy'), dl_diff_apo)

# Pipeline: freq maps with individual component -> smooth -> EB leakage correction -> NILC
## Utils for the pipeline
def smooth_tqu(map_in, lmax, beam_in, beam_out):
    # map_in should be in (3,npix)

    bl_in = hp.gauss_beam(fwhm=np.deg2rad(beam_in)/60, lmax=lmax, pol=True) # (lmax+1,4)
    bl_out = hp.gauss_beam(fwhm=np.deg2rad(beam_out)/60, lmax=lmax, pol=True)
    print(f'{bl_in.shape=}')
    alms = hp.map2alm(map_in, lmax)
    sm_alm = np.asarray([hp.almxfl(alm, bl_out[:,i]/bl_in[:,i]) for i, alm in enumerate(alms)])
    print(f'{sm_alm.shape=}')

    map_out = hp.alm2map(sm_alm, nside=nside)
    return map_out

## pipeline main function
def pp_bias(maps_in, lmax_base, beam_base, rlz_idx):
    """
        maps_in: dim=(n_maps, n_pix)
    """
    print(f"begin pipeline for bias")
    # load slope for eblc
    slope = np.load(f'../155GHz/slope_eblc/pcfn/{rlz_idx}.npy')
    print(f"eblc slope loaded")
    # load all maps
    mask_smooth = np.load(f'../../psfit/fitv4/fit_res/2048/ps_mask/new_mask/apo_C1_3.npy')
    mask_eblc = np.load(f'../../psfit/fitv4/fit_res/2048/ps_mask/new_mask/BIN_C1_3.npy')
    print(f"mask for smooth and eblc loaded")

    m_freq_list = []
    for m, freq, beam, lmax in zip(maps_in, freq_list, beam_list, lmax_list):
        print(f"process on {freq=} map:{beam=} {lmax=}")
        # 1. smooth
        smooth_m = smooth_tqu(map_in=m, lmax=lmax, beam_in=beam, beam_out=beam_base)
        print(f"smooth done!")
        # 2. EB Leakage correction
        obj_eblc = EBLeakageCorrection(m=smooth_m, lmax=lmax, nside=nside, mask=mask_eblc, post_mask=mask_eblc, slope_in=slope)
        _, _, m_freq = obj_eblc.run_eblc()
        print(f"EB Leakage correction done")

        m_freq_list.append(m_freq)

    m_freq_arr = np.asarray(m_freq_list)

    # 3. do nilc
    print(f"begin nilc")
    mask_nilc = np.load(f'../../psfit/fitv4/fit_res/2048/ps_mask/new_mask/apo_C1_3_apo_3.npy')
    obj_nilc = NILC(bandinfo='./band_info.csv', needlet_config='./needlets/0.csv', weights_config=f'./weight/std/rmv/{rlz_idx}.npz', Sm_maps=m_freq_arr, mask=mask_nilc, lmax=lmax_base, nside=nside, n_iter=3, weight_in_alm=False)
    m_bias = obj_nilc.run_nilc()
    print(f"nilc done!")

    return m_bias


def save_bias_fg():
    fgs = np.asarray([np.load(f'../../fitdata/2048/FG/{freq}/fg.npy') for freq in freq_list])

    m_fg = pp_bias(fgs, lmax_base=lmax, beam_base=beam_base, rlz_idx=rlz_idx)

    path_bias = Path(f'./data_bias/fg_rmv')
    path_bias.mkdir(parents=True, exist_ok=True)

    np.save(path_bias / Path(f'{rlz_idx}.npy'), m_fg)

def save_bias_noise():
    noises = np.asarray([gen_noise(freq, rlz_idx=rlz_idx) for freq in freq_list])

    m_noise = pp_bias(noises, lmax_base=lmax, beam_base=beam_base, rlz_idx=rlz_idx)

    path_bias = Path(f'./data_bias/noise_rmv')
    path_bias.mkdir(parents=True, exist_ok=True)

    np.save(path_bias / Path(f'{rlz_idx}.npy'), m_noise)


def save_bias_ps():
    ps = np.asarray([np.load(f'../../fitdata/2048/PS/{freq}/ps.npy') for freq in freq_list])

    m_ps = pp_bias(ps, lmax_base=lmax, beam_base=beam_base, rlz_idx=rlz_idx)

    path_bias = Path(f'./data_bias/ps_rmv')
    path_bias.mkdir(parents=True, exist_ok=True)

    np.save(path_bias / Path(f'{rlz_idx}.npy'), m_ps)

def save_bias_unresolved_ps():
    # unresolved_ps = np.asarray([np.load(f'../../fitdata/2048/PS/{freq}/ps.npy') - np.load(f'../{freq}GHz/data/ps/resolved_ps.npy') for freq in freq_list])
    unresolved_ps = np.asarray([np.load(f'../{freq}GHz/data/ps/unresolved_ps.npy') for freq in freq_list])
    print(f"{unresolved_ps.shape=}")

    m_ps = pp_bias(unresolved_ps, lmax_base=lmax, beam_base=beam_base, rlz_idx=rlz_idx)
    path_bias = Path(f'./data_bias/unresolved_ps_rmv')
    path_bias.mkdir(parents=True, exist_ok=True)

    np.save(path_bias / Path(f'{rlz_idx}.npy'), m_ps)

def save_bias_rmv_model():
    def get_m_rmv_bias(freq, beam):
        pcfn = gen_pcfn(freq=freq, beam=beam, mode='std', rlz_idx=rlz_idx)
        rmv_q = np.load(f'../{freq}GHz/fit_res/std/3sigma/map_q_{rlz_idx}.npy').copy()
        rmv_u = np.load(f'../{freq}GHz/fit_res/std/3sigma/map_u_{rlz_idx}.npy').copy()
        resolved_ps = np.load(f'../{freq}GHz/data/ps/resolved_ps.npy')
        print(f"loaded {freq} rmv bias")
        return np.asarray([np.zeros_like(rmv_q), pcfn[1].copy() - rmv_q - resolved_ps[1].copy(), pcfn[2].copy() - rmv_u - resolved_ps[2].copy()])

    rmv_bias = np.asarray([get_m_rmv_bias(freq, beam) for freq, beam in zip(freq_list, beam_list)])
    print(f"{rmv_bias.shape=}")

    m_rmv_bias = pp_bias(rmv_bias, lmax_base=lmax, beam_base=beam_base, rlz_idx=rlz_idx)
    path_bias = Path(f'./data_bias/rmv_bias_rmv')
    path_bias.mkdir(parents=True, exist_ok=True)

    np.save(path_bias / Path(f'{rlz_idx}.npy'), m_rmv_bias)

def save_bias_cfn():
    cfn = np.asarray([gen_cfn(freq=freq, beam=beam, mode='std', rlz_idx=rlz_idx) for freq, beam in zip(freq_list, beam_list)])

    m_cfn = pp_bias(cfn, lmax_base=lmax, beam_base=beam_base, rlz_idx=rlz_idx)
    path_bias = Path(f'./data_bias/cfn')
    path_bias.mkdir(parents=True, exist_ok=True)

    np.save(path_bias / Path(f'{rlz_idx}.npy'), cfn)


def check_fg_bias():
    fg = np.load(f'./data_bias/fg/{rlz_idx}.npy')
    fg1 = np.load(f'./data_bias/fg/1.npy')
    hp.orthview(fg, rot=[100,50,0], half_sky=True)
    hp.orthview(fg1, rot=[100,50,0], half_sky=True)
    hp.orthview(fg1 - fg, rot=[100,50,0], half_sky=True)
    plt.show()



## calc all bias's power spectrum
def calc_fg_bias_cl():
    # get the foreground bias upon different masks
    # that might be wrong because cmb might be a little different from ilc befored maps !!!
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam_base)/60, lmax=lmax, pol=True)[:,2]
    l_min_edges, l_max_edges = generate_bins(l_min_start=42, delta_l_min=40, l_max=lmax+1, fold=0.1, l_threshold=400)
    # delta_ell = 30
    # bin_dl = nmt.NmtBin.from_nside_linear(nside, nlb=delta_ell, is_Dell=True)
    # bin_dl = nmt.NmtBin.from_lmax_linear(lmax=lmax, nlb=40, is_Dell=True)
    bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    ell_arr = bin_dl.get_effective_ells()

    diff = np.load(f'./data_bias/fg_rmv/{rlz_idx}.npy')

    ps_mask = np.load(f'./ps_mask/union.npy')

    # hp.orthview(ps_mask, rot=[100,50,0], half_sky=True)
    # hp.orthview(apo_mask, rot=[100,50,0], half_sky=True)
    # plt.show()


    dl_diff_apo = calc_dl_from_scalar_map_bl(scalar_map=diff, apo_mask=apo_mask, bl=bl, bin_dl=bin_dl, masked_on_input=False)
    dl_diff_ps_mask = calc_dl_from_scalar_map_bl(scalar_map=diff, apo_mask=ps_mask, bl=bl, bin_dl=bin_dl, masked_on_input=False)

    path_dl_diff = Path(f'./dl_res5/fg_rmv')
    path_dl_diff.mkdir(parents=True, exist_ok=True)

    np.save(path_dl_diff / Path(f'union_{rlz_idx}.npy'), dl_diff_ps_mask)
    np.save(path_dl_diff / Path(f'apo_{rlz_idx}.npy'), dl_diff_apo)

def calc_noise_bias_cl():
    # get the foreground bias upon different masks
    # that might be wrong because cmb might be a little different from ilc befored maps !!!
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam_base)/60, lmax=lmax, pol=True)[:,2]
    l_min_edges, l_max_edges = generate_bins(l_min_start=42, delta_l_min=40, l_max=lmax+1, fold=0.1, l_threshold=400)
    # delta_ell = 30
    # bin_dl = nmt.NmtBin.from_nside_linear(nside, nlb=delta_ell, is_Dell=True)
    # bin_dl = nmt.NmtBin.from_lmax_linear(lmax=lmax, nlb=40, is_Dell=True)
    bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    ell_arr = bin_dl.get_effective_ells()

    diff = np.load(f'./data_bias/noise_rmv/{rlz_idx}.npy')

    ps_mask = np.load(f'./ps_mask/union.npy')

    # hp.orthview(ps_mask, rot=[100,50,0], half_sky=True)
    # hp.orthview(apo_mask, rot=[100,50,0], half_sky=True)
    # plt.show()


    dl_diff_apo = calc_dl_from_scalar_map_bl(scalar_map=diff, apo_mask=apo_mask, bl=bl, bin_dl=bin_dl, masked_on_input=False)
    dl_diff_ps_mask = calc_dl_from_scalar_map_bl(scalar_map=diff, apo_mask=ps_mask, bl=bl, bin_dl=bin_dl, masked_on_input=False)

    path_dl_diff = Path(f'./dl_res5/noise_rmv')
    path_dl_diff.mkdir(parents=True, exist_ok=True)

    np.save(path_dl_diff / Path(f'union_{rlz_idx}.npy'), dl_diff_ps_mask)
    np.save(path_dl_diff / Path(f'apo_{rlz_idx}.npy'), dl_diff_apo)


def calc_ps_bias_cl():
    # get the foreground bias upon different masks
    # that might be wrong because cmb might be a little different from ilc befored maps !!!
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam_base)/60, lmax=lmax, pol=True)[:,2]
    l_min_edges, l_max_edges = generate_bins(l_min_start=42, delta_l_min=40, l_max=lmax+1, fold=0.1, l_threshold=400)
    # delta_ell = 30
    # bin_dl = nmt.NmtBin.from_nside_linear(nside, nlb=delta_ell, is_Dell=True)
    # bin_dl = nmt.NmtBin.from_lmax_linear(lmax=lmax, nlb=40, is_Dell=True)
    bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    ell_arr = bin_dl.get_effective_ells()

    diff = np.load(f'./data_bias/ps_rmv/{rlz_idx}.npy')

    ps_mask = np.load(f'./ps_mask/union.npy')

    # hp.orthview(ps_mask, rot=[100,50,0], half_sky=True)
    # hp.orthview(apo_mask, rot=[100,50,0], half_sky=True)
    # plt.show()


    dl_diff_apo = calc_dl_from_scalar_map_bl(scalar_map=diff, apo_mask=apo_mask, bl=bl, bin_dl=bin_dl, masked_on_input=False)
    dl_diff_ps_mask = calc_dl_from_scalar_map_bl(scalar_map=diff, apo_mask=ps_mask, bl=bl, bin_dl=bin_dl, masked_on_input=False)

    path_dl_diff = Path(f'./dl_res5/ps_rmv')
    path_dl_diff.mkdir(parents=True, exist_ok=True)

    np.save(path_dl_diff / Path(f'union_{rlz_idx}.npy'), dl_diff_ps_mask)
    np.save(path_dl_diff / Path(f'apo_{rlz_idx}.npy'), dl_diff_apo)

def calc_unresolved_ps_bias_cl():
    # get the foreground bias upon different masks
    # that might be wrong because cmb might be a little different from ilc befored maps !!!
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam_base)/60, lmax=lmax, pol=True)[:,2]
    l_min_edges, l_max_edges = generate_bins(l_min_start=42, delta_l_min=40, l_max=lmax+1, fold=0.1, l_threshold=400)
    # delta_ell = 30
    # bin_dl = nmt.NmtBin.from_nside_linear(nside, nlb=delta_ell, is_Dell=True)
    # bin_dl = nmt.NmtBin.from_lmax_linear(lmax=lmax, nlb=40, is_Dell=True)
    bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    ell_arr = bin_dl.get_effective_ells()

    diff = np.load(f'./data_bias/unresolved_ps_rmv/{rlz_idx}.npy')

    ps_mask = np.load(f'./ps_mask/union.npy')

    # hp.orthview(ps_mask, rot=[100,50,0], half_sky=True)
    # hp.orthview(apo_mask, rot=[100,50,0], half_sky=True)
    # plt.show()


    dl_diff_apo = calc_dl_from_scalar_map_bl(scalar_map=diff, apo_mask=apo_mask, bl=bl, bin_dl=bin_dl, masked_on_input=False)
    dl_diff_ps_mask = calc_dl_from_scalar_map_bl(scalar_map=diff, apo_mask=ps_mask, bl=bl, bin_dl=bin_dl, masked_on_input=False)

    path_dl_diff = Path(f'./dl_res5/unresolved_ps_rmv')
    path_dl_diff.mkdir(parents=True, exist_ok=True)

    np.save(path_dl_diff / Path(f'union_{rlz_idx}.npy'), dl_diff_ps_mask)
    np.save(path_dl_diff / Path(f'apo_{rlz_idx}.npy'), dl_diff_apo)

def calc_rmv_bias_cl():
    # get the foreground bias upon different masks
    # that might be wrong because cmb might be a little different from ilc befored maps !!!
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam_base)/60, lmax=lmax, pol=True)[:,2]
    l_min_edges, l_max_edges = generate_bins(l_min_start=42, delta_l_min=40, l_max=lmax+1, fold=0.1, l_threshold=400)
    # delta_ell = 30
    # bin_dl = nmt.NmtBin.from_nside_linear(nside, nlb=delta_ell, is_Dell=True)
    # bin_dl = nmt.NmtBin.from_lmax_linear(lmax=lmax, nlb=40, is_Dell=True)
    bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    ell_arr = bin_dl.get_effective_ells()

    diff = np.load(f'./data_bias/rmv_bias_rmv/{rlz_idx}.npy')

    ps_mask = np.load(f'./ps_mask/union.npy')

    # hp.orthview(ps_mask, rot=[100,50,0], half_sky=True)
    # hp.orthview(apo_mask, rot=[100,50,0], half_sky=True)
    # plt.show()


    dl_diff_apo = calc_dl_from_scalar_map_bl(scalar_map=diff, apo_mask=apo_mask, bl=bl, bin_dl=bin_dl, masked_on_input=False)
    dl_diff_ps_mask = calc_dl_from_scalar_map_bl(scalar_map=diff, apo_mask=ps_mask, bl=bl, bin_dl=bin_dl, masked_on_input=False)

    path_dl_diff = Path(f'./dl_res5/rmv_bias_rmv')
    path_dl_diff.mkdir(parents=True, exist_ok=True)

    np.save(path_dl_diff / Path(f'union_{rlz_idx}.npy'), dl_diff_ps_mask)
    np.save(path_dl_diff / Path(f'apo_{rlz_idx}.npy'), dl_diff_apo)


## tests
def test_each_freq():
    freq = 215
    df = pd.read_csv(f'../{freq}GHz/mask/{freq}_after_filter.csv')

    ps = np.load(f'../../fitdata/2048/PS/{freq}/ps.npy')
    resolved_ps = np.load(f'../{freq}GHz/data/ps/resolved_ps.npy')
    for flux_idx in [0,1]:
        lon = np.rad2deg(df.at[flux_idx, 'lon'])
        lat = np.rad2deg(df.at[flux_idx, 'lat'])
        hp.gnomview(resolved_ps[1], rot=[lon,lat,0], title=f'resolved {flux_idx=}')
        hp.gnomview(ps[1], rot=[lon,lat,0], title=f'ps {flux_idx=}')
        plt.show()

def test_nilc_res():
    freq = 30
    df = pd.read_csv(f'../{freq}GHz/mask/{freq}_after_filter.csv')

    # unresolved_ps = np.load(f'./data_bias/unresolved_ps_rmv/{rlz_idx}.npy')
    # rmv_bias = np.load(f'./data_bias/rmv_bias_rmv/{rlz_idx}.npy')

    # m = np.load(f'./data2/std/pcfn/0.npy')
    m = np.load(f'./data_bias/ps/0.npy')
    mask = np.load(f"./ps_mask/union.npy")

    # hp.orthview(unresolved_ps, rot=[100,50,0], title='unresolved ps')
    # hp.orthview(rmv_bias, rot=[100,50,0], title='rmv bias')
    hp.orthview(m, rot=[100,50,0], title='B mode')
    
    plt.show()
    for flux_idx in [2,6]:
        lon = np.rad2deg(df.at[flux_idx, 'lon'])
        lat = np.rad2deg(df.at[flux_idx, 'lat'])
        # hp.gnomview(unresolved_ps, rot=[lon,lat,0], title=f'unresolved ps {flux_idx=} B')
        # hp.gnomview(rmv_bias, rot=[lon,lat,0], title=f'rmv bias {flux_idx=} B')
        hp.gnomview(m, rot=[lon,lat,0], title=f'{flux_idx=} B mode')
        hp.gnomview(m*mask, rot=[lon,lat,0], title=f'{flux_idx=} B mode mask')

        plt.show()

def gen_apodized_ps_mask():
    ori_mask = np.load(f"../../psfit/fitv4/fit_res/2048/ps_mask/new_mask/apo_C1_3_apo_3_apo_3.npy")
    freq = 30
    beam = 67
    # df = pd.read_csv(f'../{freq}GHz/mask/{freq}_after_filter.csv')
    # df = pd.read_csv(f'./concat_ps.csv')
    df = pd.read_csv(f'../30GHz/mask/30_after_filter.csv')
    mask = np.ones(hp.nside2npix(nside))
    for flux_idx in range(len(df)):
        print(f'{flux_idx=}')
        # if flux_idx > 0:
            # break
        lon = np.rad2deg(df.at[flux_idx, 'lon'])
        lat = np.rad2deg(df.at[flux_idx, 'lat'])

        ctr_vec = hp.ang2vec(theta=lon, phi=lat, lonlat=True)
        ipix_mask = hp.query_disc(nside=nside, vec=ctr_vec, radius=2.5 * np.deg2rad(beam) / 60)
        mask[ipix_mask] = 0

        # fig_size=200
        # # hp.gnomview(ori_mask, rot=[lon, lat, 0], title='before mask', xsize=fig_size)
        # hp.gnomview(mask, rot=[lon, lat, 0], title='after mask', xsize=fig_size)
        # plt.show()

    # apo_mask = nmt.mask_apodization(mask_in=mask, aposize=1)
    # hp.orthview(apo_mask * ori_mask, rot=[100,50, 0], title='mask', xsize=2000)
    # plt.show()

    path_mask = Path('./ps_mask')
    path_mask.mkdir(exist_ok=True, parents=True)
    # np.save(f'./ps_mask/{freq}GHz_1deg.npy', apo_mask * ori_mask)
    # np.save(f'./ps_mask/{freq}GHz.npy', mask * ori_mask)
    np.save(f'./ps_mask/30GHz_67.npy', mask * ori_mask)

def test_check_map():
    m1 = np.load('./ps_mask/30GHz.npy')
    m2 = np.load('./ps_mask/30GHz_67.npy')
    hp.orthview(m1, rot=[100,50,0], half_sky=True)
    hp.orthview(m2, rot=[100,50,0], half_sky=True)
    plt.show()

def test_deconvolve():
    flux_idx = 0
    def see_map(m):
        #see the map
        df = pd.read_csv(f'../30GHz/mask/30_after_filter.csv')
        lon = np.rad2deg(df.at[flux_idx, 'lon'])
        lat = np.rad2deg(df.at[flux_idx, 'lat'])
        hp.gnomview(m, rot=[lon, lat, 0])
        plt.show()

    mask_17 = np.load(f"./ps_mask/30GHz_one_ps_17.npy")
    mask_67 = np.load(f"./ps_mask/30GHz_one_ps_67.npy")
    see_map(mask_17)
    see_map(mask_67)

    m_freq = np.load(f'../30GHz/data/ps/30GHz_one_ps.npy')
    sm_freq = smooth_tqu(map_in=m_freq, lmax=500, beam_in=67, beam_out=17)
    m_b_freq = hp.alm2map(hp.map2alm(sm_freq, lmax=500)[2], nside=nside)
    np.save(f'./test/m_b_freq.npy', m_b_freq)
    see_map(m_b_freq)
    see_map(m_b_freq*mask_17)
    see_map(m_b_freq*mask_67)

if __name__ == "__main__":
    # calc_eblc_bias()

    # save_bias_fg()
    # save_bias_ps()
    # save_bias_unresolved_ps()
    # save_bias_rmv_model()
    # save_bias_cfn()
    # save_bias_noise()

    # check_fg_bias()

    # calc_fg_bias_cl()
    # calc_ps_bias_cl()
    # calc_unresolved_ps_bias_cl()
    # calc_rmv_bias_cl()
    calc_noise_bias_cl()

    # test_each_freq()
    # test_nilc_res()
    # test_check_map()
    # gen_apodized_ps_mask()
    # test_deconvolve()


