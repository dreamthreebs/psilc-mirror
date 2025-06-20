import numpy as np
import healpy as hp
import pandas as pd
import time
import matplotlib.pyplot as plt
import pymaster as nmt
from pathlib import Path
from nilc import NILC
rlz_idx = 0
freq_list = [30, 95, 155, 215, 270]
beam_list = [67, 30, 17, 11, 9]
beam_base = 17 #arcmin
lmax = 1500
nside = 2048

cmb_seed = np.load('../seeds_cmb_2k.npy')
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

def gen_map(rlz_idx=0, mode='mean', return_noise=False):
    # mode can be mean or std
    nside = 2048

    nstd = np.load(f'../../../FGSim/NSTDNORTH/2048/{freq}.npy')
    npix = hp.nside2npix(nside=2048)
    np.random.seed(seed=noise_seed[rlz_idx])
    # noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))
    noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))
    print(f"{np.std(noise[1])=}")

    if return_noise:
        return noise

    ps = np.load(f'../../../fitdata/2048/PS/{freq}/ps.npy')
    fg = np.load(f'../../../fitdata/2048/FG/{freq}/fg.npy')

    cls = np.load('../../../src/cmbsim/cmbdata/cmbcl_8k.npy')
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

def gen_cmb(beam, rlz_idx=0):

    cls = np.load('../../src/cmbsim/cmbdata/cmbcl_8k.npy')
    np.random.seed(seed=cmb_seed[rlz_idx])

    cmb_iqu = hp.synfast(cls.T, nside=nside, fwhm=np.deg2rad(beam)/60, new=True, lmax=3*nside-1)
    return cmb_iqu


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
    method = 'cf'
    cf = np.asarray([np.load(f'../{freq}GHz/fit_res/sm_new/{sim_mode}/{method}/{rlz_idx}.npy') for freq in freq_list])

    obj_cf = NILC(bandinfo='./band_info.csv', needlet_config='./needlets/0.csv', weights_config=f'./weight/std/pcfn/{rlz_idx}.npz', Sm_maps=cf, mask=ilc_mask, lmax=lmax, nside=nside, n_iter=3, weight_in_alm=False)
    cln_cf = obj_cf.run_nilc()

    Path(f'./data3/cf').mkdir(exist_ok=True, parents=True)
    np.save(f'./data3/cf/{rlz_idx}.npy', cln_cf)

def check_fg_bias():
    cln_cf = np.load(f'./data3/cf/{rlz_idx}.npy')
    fid_c = np.load(f'./data3/fid_c/{rlz_idx}.npy')

    # hp.orthview(cln_cf, rot=[100,50,0], half_sky=True, min=-0.7, max=0.7)
    # hp.orthview(fid_c, rot=[100,50,0], half_sky=True, min=-0.7, max=0.7)
    # hp.orthview(fid_c - cln_cf, rot=[100,50,0], half_sky=True, min=-0.7, max=0.7)
    hp.orthview((fid_c - cln_cf) * apo_mask, rot=[100,50,0], half_sky=True)
    plt.show()

def calc_fg_bias_cl():
    pass
    


if __name__ == "__main__":
    # calc_eblc_bias()
    # calc_fg_bias()
    check_fg_bias()

