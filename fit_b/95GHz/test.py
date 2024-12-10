import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pymaster as nmt
import os,sys
import pandas as pd

from scipy.interpolate import interp1d
from pathlib import Path
from config import lmax, nside, freq, beam
from pix_cov_qu import CovCalculator
from fit_qu_no_const import FitPolPS

df = pd.read_csv(f'./mask/{freq}.csv')
ori_mask = np.load('../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5.npy')
ori_apo_mask = np.load('../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5APO_5.npy')
n_ps_list = np.linspace(10, 135, 8, dtype=int)
print(f'{n_ps_list=}')

noise_seeds = np.load('../seeds_noise_2k.npy')
cmb_seeds = np.load('../seeds_cmb_2k.npy')
fg_seeds = np.load('../seeds_fg_2k.npy')

rlz_idx = 0

# Utils
def dl2cl(D_ell):

    ell = np.arange(len(D_ell))
    mask = ell > 1
    C_ell = np.zeros_like(D_ell, dtype=np.float64)
    C_ell[mask] = (2 * np.pi * D_ell[mask]) / (ell[mask] * (ell[mask] + 1))
    C_ell[~mask] = 0
    return C_ell

# Part 1: Generate and check point sources mask
def gen_mask():
    # masked point sources from 10-135
    radius_factor = 1.5
    Path('./ps_mask').mkdir(exist_ok=True, parents=True)
    for n_ps in n_ps_list:
        mask = np.ones(hp.nside2npix(nside))
        print(f'{n_ps=}')
        for flux_idx in range(n_ps):
            lon = np.rad2deg(df.at[flux_idx, 'lon'])
            lat = np.rad2deg(df.at[flux_idx, 'lat'])
            ctr_vec = hp.ang2vec(theta=lon, phi=lat, lonlat=True)
            ipix_mask = hp.query_disc(nside=nside, vec=ctr_vec, radius=radius_factor * np.deg2rad(beam) / 60)
            mask[ipix_mask] = 0

        apo_ps_mask = nmt.mask_apodization(mask, aposize=1) * ori_apo_mask
        np.save(f'./ps_mask/{n_ps}_ps_mask.npy', apo_ps_mask)

def check_mask():
    for n_ps in n_ps_list:
        mask = np.load(f'./ps_mask/{n_ps}_ps_mask.npy')
        hp.orthview(mask, rot=[100,50,0], half_sky=True)
        plt.show()

# Part 2: Estimate powerspectrum
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

def calc_dl_from_pol_map(m_q, m_u, bl, apo_mask, bin_dl, masked_on_input, purify_b):
    f2p = nmt.NmtField(apo_mask, [m_q, m_u], beam=bl, masked_on_input=masked_on_input, purify_b=purify_b, lmax=lmax, lmax_mask=lmax)
    w22p = nmt.NmtWorkspace.from_fields(f2p, f2p, bin_dl)
    # dl = nmt.workspaces.compute_full_master(pol_field, pol_field, b=bin_dl)
    dl = w22p.decouple_cell(nmt.compute_coupled_cell(f2p, f2p))[3]
    return dl

def gen_fg_cl():
    cl_fg = np.load('./data/debeam_full_b/cl_fg.npy')
    Cl_TT = cl_fg[0]
    Cl_EE = cl_fg[1]
    Cl_BB = cl_fg[2]
    Cl_TE = np.zeros_like(Cl_TT)
    return np.array([Cl_TT, Cl_EE, Cl_BB, Cl_TE])

def gen_map(rlz_idx):
    npix = hp.nside2npix(nside=nside)
    ps = np.load('./data/ps/ps.npy')

    nstd = np.load(f'../../FGSim/NSTDNORTH/2048/{freq}.npy')
    np.random.seed(seed=noise_seeds[rlz_idx])
    # noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))
    noise = nstd * np.random.normal(loc=0, scale=1, size=(3,npix))
    print(f"{np.std(noise[1])=}")

    # cmb_iqu = np.load(f'../../fitdata/2048/CMB/215/{rlz_idx}.npy')
    # cls = np.load('../../src/cmbsim/cmbdata/cmbcl.npy')
    cls = np.load('../../src/cmbsim/cmbdata/cmbcl_8k.npy')
    np.random.seed(seed=cmb_seeds[rlz_idx])
    # cmb_iqu = hp.synfast(cls.T, nside=nside, fwhm=np.deg2rad(beam)/60, new=True, lmax=1999)
    cmb_iqu = hp.synfast(cls.T, nside=nside, fwhm=np.deg2rad(beam)/60, new=True, lmax=lmax)

    cls_fg = gen_fg_cl()
    np.random.seed(seed=fg_seeds[rlz_idx])
    fg = hp.synfast(cls_fg, nside=nside, fwhm=0, new=True, lmax=lmax)

    pcfn = noise + ps + cmb_iqu + fg
    cfn = noise + cmb_iqu + fg
    cf = cmb_iqu + fg
    n = noise

    # pcfn = noise
    # cfn = noise
    # cf = noise
    # n = noise

    return pcfn, cfn, cf, n

def gen_noise(rlz_idx):
    npix = hp.nside2npix(nside=nside)

    nstd = np.load(f'../../FGSim/NSTDNORTH/2048/{freq}.npy')
    np.random.seed(seed=noise_seeds[rlz_idx])
    # noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))
    noise = nstd * np.random.normal(loc=0, scale=1, size=(3,npix))
    print(f"{np.std(noise[1])=}")

    return noise


def estimate_powerspectrum():
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax, pol=True)[:,2]
    l_min_edges, l_max_edges = generate_bins(l_min_start=30, delta_l_min=30, l_max=lmax+1, fold=0.2)
    # delta_ell = 30
    # bin_dl = nmt.NmtBin.from_nside_linear(nside, nlb=delta_ell, is_Dell=True)
    # bin_dl = nmt.NmtBin.from_lmax_linear(lmax=lmax, nlb=30, is_Dell=True)
    bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)

    ell_arr = bin_dl.get_effective_ells()

    m_pcfn, m_cfn, m_cf, m_n= gen_map(rlz_idx=rlz_idx)
    for n_ps in n_ps_list:
        print(f'{n_ps=}')
        mask = np.load(f'./ps_mask/{n_ps}_ps_mask.npy')

        dl_pcfn = calc_dl_from_pol_map(m_q=m_pcfn[1].copy()*ori_mask, m_u=m_pcfn[2].copy()*ori_mask, bl=bl, apo_mask=mask, bin_dl=bin_dl, masked_on_input=False, purify_b=True)
        dl_cfn = calc_dl_from_pol_map(m_q=m_cfn[1].copy()*ori_mask, m_u=m_cfn[2].copy()*ori_mask, bl=bl, apo_mask=mask, bin_dl=bin_dl, masked_on_input=False, purify_b=True)
        dl_cf = calc_dl_from_pol_map(m_q=m_cf[1].copy()*ori_mask, m_u=m_cf[2].copy()*ori_mask, bl=bl, apo_mask=mask, bin_dl=bin_dl, masked_on_input=False, purify_b=True)
        dl_n = calc_dl_from_pol_map(m_q=m_n[1].copy()*ori_mask, m_u=m_n[2].copy()*ori_mask, bl=bl, apo_mask=mask, bin_dl=bin_dl, masked_on_input=False, purify_b=True)

        path_dl_pcfn = Path(f'Dl_res/{n_ps}_ps_mask/pcfn')
        path_dl_pcfn.mkdir(exist_ok=True, parents=True)
        path_dl_cfn = Path(f'Dl_res/{n_ps}_ps_mask/cfn')
        path_dl_cfn.mkdir(exist_ok=True, parents=True)
        path_dl_cf = Path(f'Dl_res/{n_ps}_ps_mask/cf')
        path_dl_cf.mkdir(exist_ok=True, parents=True)
        path_dl_n = Path(f'Dl_res/{n_ps}_ps_mask/n')
        path_dl_n.mkdir(exist_ok=True, parents=True)


        np.save(path_dl_pcfn / Path(f'{rlz_idx}.npy'), dl_pcfn)
        np.save(path_dl_cfn / Path(f'{rlz_idx}.npy'), dl_cfn)
        np.save(path_dl_cf / Path(f'{rlz_idx}.npy'), dl_cf)
        np.save(path_dl_n / Path(f'{rlz_idx}.npy'), dl_n)

def estimate_powerspectrum_ori():
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax, pol=True)[:,2]
    l_min_edges, l_max_edges = generate_bins(l_min_start=30, delta_l_min=30, l_max=lmax+1, fold=0.2)
    # delta_ell = 30
    # bin_dl = nmt.NmtBin.from_nside_linear(nside, nlb=delta_ell, is_Dell=True)
    # bin_dl = nmt.NmtBin.from_lmax_linear(lmax=lmax, nlb=30, is_Dell=True)
    bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)

    ell_arr = bin_dl.get_effective_ells()

    m_pcfn, m_cfn, m_cf, m_n= gen_map(rlz_idx=rlz_idx)
    n_ps = 10
    print(f'{n_ps=}')

    dl_pcfn = calc_dl_from_pol_map(m_q=m_pcfn[1].copy()*ori_mask, m_u=m_pcfn[2].copy()*ori_mask, bl=bl, apo_mask=ori_apo_mask, bin_dl=bin_dl, masked_on_input=False, purify_b=True)
    dl_cfn = calc_dl_from_pol_map(m_q=m_cfn[1].copy()*ori_mask, m_u=m_cfn[2].copy()*ori_mask, bl=bl, apo_mask=ori_apo_mask, bin_dl=bin_dl, masked_on_input=False, purify_b=True)
    dl_cf = calc_dl_from_pol_map(m_q=m_cf[1].copy()*ori_mask, m_u=m_cf[2].copy()*ori_mask, bl=bl, apo_mask=ori_apo_mask, bin_dl=bin_dl, masked_on_input=False, purify_b=True)
    dl_n = calc_dl_from_pol_map(m_q=m_n[1].copy()*ori_mask, m_u=m_n[2].copy()*ori_mask, bl=bl, apo_mask=ori_apo_mask, bin_dl=bin_dl, masked_on_input=False, purify_b=True)


    path_dl_pcfn = Path(f'Dl_res/{n_ps}_ori_mask/pcfn')
    path_dl_pcfn.mkdir(exist_ok=True, parents=True)
    path_dl_cfn = Path(f'Dl_res/{n_ps}_ori_mask/cfn')
    path_dl_cfn.mkdir(exist_ok=True, parents=True)
    path_dl_cf = Path(f'Dl_res/{n_ps}_ori_mask/cf')
    path_dl_cf.mkdir(exist_ok=True, parents=True)
    path_dl_n = Path(f'Dl_res/{n_ps}_ori_mask/n')
    path_dl_n.mkdir(exist_ok=True, parents=True)

    np.save(path_dl_pcfn / Path(f'{rlz_idx}.npy'), dl_pcfn)
    np.save(path_dl_cfn / Path(f'{rlz_idx}.npy'), dl_cfn)
    np.save(path_dl_cf / Path(f'{rlz_idx}.npy'), dl_cf)
    np.save(path_dl_n / Path(f'{rlz_idx}.npy'), dl_n)


def check_powerspectrum_ps_mask():
    # bin_dl = nmt.NmtBin.from_lmax_linear(lmax=lmax, nlb=4, is_Dell=True)
    # ell_arr = bin_dl.get_effective_ells()

    l_min_edges, l_max_edges = generate_bins(l_min_start=30, delta_l_min=30, l_max=lmax+1, fold=0.2)
    # delta_ell = 30
    # bin_dl = nmt.NmtBin.from_nside_linear(nside, nlb=delta_ell, is_Dell=True)
    # bin_dl = nmt.NmtBin.from_lmax_linear(lmax=lmax, nlb=30, is_Dell=True)
    bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    ell_arr = bin_dl.get_effective_ells()



    rlz_idx = 0
    n_ps = 10
    dl_pcfn_mean_list = []
    dl_cfn_mean_list = []
    dl_cf_mean_list = []
    dl_n_mean_list = []

    dl_pcfn_std_list = []
    dl_cfn_std_list = []
    dl_cf_std_list = []
    dl_n_std_list = []

    for n_ps in n_ps_list:
        print(f'{n_ps=}')
        dl_pcfn_list = []
        dl_cfn_list = []
        dl_cf_list = []
        dl_n_list = []
        for rlz_idx in range(200):
            dl_n = np.load(f'./Dl_res/{n_ps}_ps_mask/n/{rlz_idx}.npy')
            dl_pcfn = np.load(f'./Dl_res/{n_ps}_ps_mask/pcfn/{rlz_idx}.npy') - dl_n
            dl_cfn = np.load(f'./Dl_res/{n_ps}_ps_mask/cfn/{rlz_idx}.npy') - dl_n
            dl_cf = np.load(f'./Dl_res/{n_ps}_ps_mask/cf/{rlz_idx}.npy')
            dl_pcfn_list.append(dl_pcfn)
            dl_cfn_list.append(dl_cfn)
            dl_cf_list.append(dl_cf)
            dl_n_list.append(dl_n)

        dl_pcfn_mean = np.mean(dl_pcfn_list, axis=0)
        dl_cfn_mean = np.mean(dl_cfn_list, axis=0)
        dl_cf_mean = np.mean(dl_cf_list, axis=0)
        dl_n_mean = np.mean(dl_n_list, axis=0)

        dl_pcfn_std = np.std(dl_pcfn_list, axis=0)
        dl_cfn_std = np.std(dl_cfn_list, axis=0)
        dl_cf_std = np.std(dl_cf_list, axis=0)
        dl_n_std = np.std(dl_n_list, axis=0)

        dl_pcfn_mean_list.append(dl_pcfn_mean)
        dl_cfn_mean_list.append(dl_cfn_mean)
        dl_cf_mean_list.append(dl_cf_mean)
        dl_n_mean_list.append(dl_n_mean)

        dl_pcfn_std_list.append(dl_pcfn_std)
        dl_cfn_std_list.append(dl_cfn_std)
        dl_cf_std_list.append(dl_cf_std)
        dl_n_std_list.append(dl_n_std)


        plt.figure(1)
        plt.loglog(ell_arr, dl_pcfn_mean, label='pcfn')
        plt.loglog(ell_arr, dl_cfn_mean, label='cfn')
        plt.loglog(ell_arr, dl_cf_mean, label='cf')
        plt.loglog(ell_arr, dl_n_mean, label='n')
        plt.legend()
        plt.figure(2)
        plt.loglog(ell_arr, dl_pcfn_std, label='pcfn')
        plt.loglog(ell_arr, dl_cfn_std, label='cfn')
        plt.loglog(ell_arr, dl_cf_std, label='cf')
        plt.loglog(ell_arr, dl_n_std, label='n')
        plt.legend()
        plt.show()

    # dl_ori_pcfn_mean = np.load('./ori_pcfn_mean.npy')
    # dl_ori_cf_mean = np.load('./ori_cf_mean.npy')

    plt.figure(1)
    for i, n_ps in enumerate(n_ps_list):
        plt.loglog(ell_arr, dl_pcfn_mean_list[i], label=f'pcfn, {n_ps=}')

        # plt.loglog(ell_arr, dl_cfn_mean, label='cfn')
        # plt.loglog(ell_arr, dl_cf_mean, label='cf')
        # plt.loglog(ell_arr, dl_n_mean, label='n')

    # plt.loglog(ell_arr, dl_ori_pcfn_mean, label=f'pcfn ori')
    # plt.loglog(ell_arr, dl_ori_cf_mean, label=f'cf ori')
    plt.title('mean')
    plt.legend()

    # dl_ori_pcfn_std = np.load('./ori_pcfn_std.npy')
    # dl_ori_cf_std = np.load('./ori_cf_std.npy')

    plt.figure(2)
    for i, n_ps in enumerate(n_ps_list):
        plt.loglog(ell_arr, dl_pcfn_std_list[i], label=f'pcfn, {n_ps=}')

    # plt.loglog(ell_arr, dl_ori_pcfn_std, label=f'pcfn ori')
    # plt.loglog(ell_arr, dl_ori_cf_std, label=f'cf ori')

    plt.title('std')

    plt.legend()
    plt.show()

    # plt.figure(2)
    # plt.loglog(ell_arr, dl_pcfn_std, label='pcfn')
    # plt.loglog(ell_arr, dl_cfn_std, label='cfn')
    # plt.loglog(ell_arr, dl_cf_std, label='cf')
    # plt.loglog(ell_arr, dl_n_std, label='n')
    # plt.legend()
    # plt.show()


def check_powerspectrum_ori_mask():
    bin_dl = nmt.NmtBin.from_lmax_linear(lmax=lmax, nlb=4, is_Dell=True)
    ell_arr = bin_dl.get_effective_ells()

    rlz_idx = 0
    n_ps = 27

    dl_pcfn_list = []
    dl_cfn_list = []
    dl_cf_list = []
    dl_n_list = []
    for rlz_idx in range(200):
        dl_n = np.load(f'./Dl_res/{n_ps}_ori_mask/n/{rlz_idx}.npy')
        dl_pcfn = np.load(f'./Dl_res/{n_ps}_ori_mask/pcfn/{rlz_idx}.npy') - dl_n
        dl_cfn = np.load(f'./Dl_res/{n_ps}_ori_mask/cfn/{rlz_idx}.npy') - dl_n
        dl_cf = np.load(f'./Dl_res/{n_ps}_ori_mask/cf/{rlz_idx}.npy')
        dl_pcfn_list.append(dl_pcfn)
        dl_cfn_list.append(dl_cfn)
        dl_cf_list.append(dl_cf)
        dl_n_list.append(dl_n)

    dl_pcfn_mean = np.mean(dl_pcfn_list, axis=0)
    dl_cfn_mean = np.mean(dl_cfn_list, axis=0)
    dl_cf_mean = np.mean(dl_cf_list, axis=0)
    dl_n_mean = np.mean(dl_n_list, axis=0)
    np.save('ori_pcfn_mean.npy', dl_pcfn_mean)
    np.save('ori_cfn_mean.npy', dl_cfn_mean)
    np.save('ori_cf_mean.npy', dl_cf_mean)
    np.save('ori_n_mean.npy', dl_n_mean)

    dl_pcfn_std = np.std(dl_pcfn_list, axis=0)
    dl_cfn_std = np.std(dl_cfn_list, axis=0)
    dl_cf_std = np.std(dl_cf_list, axis=0)
    dl_n_std = np.std(dl_n_list, axis=0)
    np.save('ori_pcfn_std.npy', dl_pcfn_std)
    np.save('ori_cfn_std.npy', dl_cfn_std)
    np.save('ori_cf_std.npy', dl_cf_std)
    np.save('ori_n_std.npy', dl_n_std)



    # plt.figure(1)
    # plt.loglog(ell_arr, dl_pcfn_mean, label='pcfn')
    # plt.loglog(ell_arr, dl_cfn_mean, label='cfn')
    # plt.loglog(ell_arr, dl_cf_mean, label='cf')
    # plt.loglog(ell_arr, dl_n_mean, label='n')
    # plt.legend()

    # plt.figure(2)
    # plt.loglog(ell_arr, dl_pcfn_std, label='pcfn')
    # plt.loglog(ell_arr, dl_cfn_std, label='cfn')
    # plt.loglog(ell_arr, dl_cf_std, label='cf')
    # plt.loglog(ell_arr, dl_n_std, label='n')
    # plt.legend()

    # plt.show()

def check_powerspectrum_one_rlz():
    # bin_dl = nmt.NmtBin.from_lmax_linear(lmax=lmax, nlb=4, is_Dell=True)
    l_min_edges, l_max_edges = generate_bins(l_min_start=30, delta_l_min=30, l_max=lmax+1, fold=0.2)
    # delta_ell = 30
    # bin_dl = nmt.NmtBin.from_nside_linear(nside, nlb=delta_ell, is_Dell=True)
    # bin_dl = nmt.NmtBin.from_lmax_linear(lmax=lmax, nlb=30, is_Dell=True)
    bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)

    ell_arr = bin_dl.get_effective_ells()

    cl_fg = gen_fg_cl()
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax, pol=True)[:,2]
    cl_cmb = np.load('../../src/cmbsim/cmbdata/cmbcl_8k.npy').T[2, :lmax+1]
    l = np.arange(lmax+1)

    plt.loglog(l, l*(l+1)*(cl_fg[2]/bl**2 + cl_cmb)/(2*np.pi), label='input fg+cmb', color='black')

    rlz_idx = 0

    ori_n = np.load(f'./Dl_res/10_ori_mask/n/{rlz_idx}.npy')
    ori_pcfn = np.load(f'./Dl_res/10_ori_mask/pcfn/{rlz_idx}.npy') - ori_n
    ori_cfn = np.load(f'./Dl_res/10_ori_mask/cfn/{rlz_idx}.npy') - ori_n
    ori_cf = np.load(f'./Dl_res/10_ori_mask/cf/{rlz_idx}.npy')
    x_interp = np.arange(lmax+1)
    interp_func = interp1d(ell_arr, ori_cf, kind='cubic', fill_value='extrapolate')
    cl_fg_ori = interp_func(x_interp)

    # cl_fg_ori = bin_dl.unbin_cell(ori_cf)
    plt.loglog(l, cl_fg_ori, label=f'after interpolation')
    plt.loglog(ell_arr, ori_cf, label=f'no ps mask cf {rlz_idx=}')
    # plt.loglog(ell_arr, ori_pcfn, label=f'no ps mask pcfn {rlz_idx=}')

    for n_ps in n_ps_list:
        n = np.load(f'./Dl_res/{n_ps}_ps_mask/n/{rlz_idx}.npy')
        pcfn = np.load(f'./Dl_res/{n_ps}_ps_mask/pcfn/{rlz_idx}.npy') - n
        cfn = np.load(f'./Dl_res/{n_ps}_ps_mask/cfn/{rlz_idx}.npy') - n
        cf = np.load(f'./Dl_res/{n_ps}_ps_mask/cf/{rlz_idx}.npy')
        plt.loglog(ell_arr, pcfn, label=f'pcfn {n_ps=}')

    plt.xlabel(f'$\\ell$')
    plt.ylabel(f'$D_\\ell [\\mu K^2]$')

    plt.legend()
    plt.show()

def check_powerspectrum_rmse():
    l_min_edges, l_max_edges = generate_bins(l_min_start=30, delta_l_min=30, l_max=lmax+1, fold=0.2)
    # delta_ell = 30
    # bin_dl = nmt.NmtBin.from_nside_linear(nside, nlb=delta_ell, is_Dell=True)
    # bin_dl = nmt.NmtBin.from_lmax_linear(lmax=lmax, nlb=30, is_Dell=True)
    bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    ell_arr = bin_dl.get_effective_ells()
    print(f'{ell_arr=}')
    # bin_dl = nmt.NmtBin.from_lmax_linear(lmax=lmax, nlb=4, is_Dell=True)
    # ell_arr = bin_dl.get_effective_ells()

    rlz_idx = 0
    n_ps = 10

    ori_cf = np.load(f'./Dl_res/10_ori_mask/cf/{rlz_idx}.npy')

    dl_pcfn_rmse_list = []

    for n_ps in n_ps_list:
        print(f'{n_ps=}')
        dl_pcfn_list = []
        dl_cfn_list = []
        dl_cf_list = []
        dl_n_list = []
        for rlz_idx in range(200):
            dl_n = np.load(f'./Dl_res/{n_ps}_ps_mask/n/{rlz_idx}.npy')
            dl_pcfn = np.load(f'./Dl_res/{n_ps}_ps_mask/pcfn/{rlz_idx}.npy') - dl_n
            dl_cfn = np.load(f'./Dl_res/{n_ps}_ps_mask/cfn/{rlz_idx}.npy') - dl_n
            dl_cf = np.load(f'./Dl_res/{n_ps}_ps_mask/cf/{rlz_idx}.npy')
            dl_pcfn_list.append(dl_pcfn)
            dl_cfn_list.append(dl_cfn)
            dl_cf_list.append(dl_cf)
            dl_n_list.append(dl_n)

        dl_pcfn_rmse = np.sqrt(np.mean((np.asarray(dl_pcfn_list)-ori_cf)**2, axis=0))
        dl_pcfn_rmse_list.append(dl_pcfn_rmse)
        pcfn_rmse_ratio = np.sum(dl_pcfn_rmse[:13] / ori_cf[:13])
        print(f'{pcfn_rmse_ratio=}')

    plt.figure(1)
    for i, n_ps in enumerate(n_ps_list):
        plt.loglog(ell_arr, dl_pcfn_rmse_list[i], label=f'pcfn rmse, {n_ps=}')

    plt.title('RMSE')
    plt.xlabel(f'$\\ell$')
    plt.ylabel(f'$\\Delta D_\\ell [\\mu K^2]$')

    plt.legend()
    plt.show()

# Part 3: Interpolate estimated band power into power spectrum!!
def bandpower2powerspectrum():
    rlz_idx = 0
    l_min_edges, l_max_edges = generate_bins(l_min_start=30, delta_l_min=30, l_max=lmax+1, fold=0.2)
    bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    ell_arr = bin_dl.get_effective_ells()

    l = np.arange(lmax+1)

    for n_ps in n_ps_list:
        n = np.load(f'./Dl_res/{n_ps}_ps_mask/n/{rlz_idx}.npy')
        pcfn = np.load(f'./Dl_res/{n_ps}_ps_mask/pcfn/{rlz_idx}.npy') - n
        # plt.loglog(ell_arr, pcfn, label=f'pcfn {n_ps=}')
        interp_func = interp1d(ell_arr, pcfn, kind='cubic', fill_value='extrapolate')
        pcfn_interp = interp_func(l)
        pcfn_interp[0:2] = 0
        plt.loglog(l, pcfn_interp, label=f'pcfn interp {n_ps=}')
        path_dl_interp = Path('./Dl_interp/pcfn')
        path_dl_interp.mkdir(exist_ok=True, parents=True)
        np.save(path_dl_interp / Path(f'{n_ps}.npy'), pcfn_interp)
        # plt.show()

    plt.legend()
    plt.show()

def gen_cmb_cl(beam, lmax):
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=10000, pol=True)
    print(f'{bl[0:10,0]=}')
    print(f'{bl[0:10,1]=}')
    print(f'{bl[0:10,2]=}')
    print(f'{bl[0:10,3]=}')
    # cl = np.load('../../src/cmbsim/cmbdata/cmbcl.npy')
    cl = np.load('../../src/cmbsim/cmbdata/cmbcl_8k.npy')
    print(f'{cl.shape=}')

    Cl_TT = cl[0:lmax+1,0] * bl[0:lmax+1,0]**2
    Cl_EE = cl[0:lmax+1,1] * bl[0:lmax+1,1]**2
    Cl_BB = cl[0:lmax+1,2] * bl[0:lmax+1,2]**2
    Cl_TE = cl[0:lmax+1,3] * bl[0:lmax+1,3]**2
    return np.asarray([Cl_TT, Cl_EE, Cl_BB, Cl_TE])

def check_input_cl():
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax, pol=True)[:,2]
    l = np.arange(lmax + 1)
    cl_fg = gen_fg_cl()
    cl_cmb = gen_cmb_cl(beam=beam, lmax=lmax)

    n_ps = 81
    dl_ps_fg = np.load(f'./Dl_interp/pcfn/{n_ps}.npy')
    for i in range(3):

        plt.loglog(l, l*(l+1)*cl_fg[i]/bl**2/(2*np.pi), label='fg')
        plt.loglog(l, l*(l+1)*cl_cmb[i]/bl**2/(2*np.pi), label='cmb')
        plt.loglog(l, l*(l+1)*(cl_cmb[i]+cl_fg[i])/bl**2/(2*np.pi), label='cmb + fg')
        if i==2:
            # plt.loglog(l, dl_ps_fg*bl[:,i]**2, label='ps fg')
            plt.loglog(l[:1001], dl_ps_fg, label='ps fg')
        plt.legend()
        plt.show()

def calc_th_cov():
    cl_fg = gen_fg_cl()
    cl_cmb = gen_cmb_cl(beam=beam, lmax=lmax)

    cl_tot = cl_fg + cl_cmb

    flux_idx = 0
    pix_ind = np.load(f'./pix_idx_qu/{flux_idx}.npy')
    obj_cov = CovCalculator(nside=nside, lmin=2, lmax=lmax, Cl_TT=cl_tot[0], Cl_EE=cl_tot[1], Cl_BB=cl_tot[2], Cl_TE=cl_tot[3], pixind=pix_ind, calc_opt='polarization', out_pol_opt='QU')
    MP = obj_cov.run_calc_cov()

    path_cov = Path('./cmb_qu_cov')
    path_cov.mkdir(exist_ok=True, parents=True)
    # np.save(f'./test_class_cov/cmb.npy', MP)
    np.save(Path(path_cov / f'{flux_idx}.npy'), MP)

def calc_cmb_cov():
    cl_fg = gen_fg_cl()
    cl_cmb = gen_cmb_cl(beam=beam, lmax=lmax)

    cl_tot = cl_cmb

    flux_idx = 0
    pix_ind = np.load(f'./pix_idx_qu/{flux_idx}.npy')
    obj_cov = CovCalculator(nside=nside, lmin=2, lmax=lmax, Cl_TT=cl_tot[0], Cl_EE=cl_tot[1], Cl_BB=cl_tot[2], Cl_TE=cl_tot[3], pixind=pix_ind, calc_opt='polarization', out_pol_opt='QU')
    MP = obj_cov.run_calc_cov()

    path_cov = Path('./cmb_qu_cov')
    path_cov.mkdir(exist_ok=True, parents=True)
    # np.save(f'./test_class_cov/cmb.npy', MP)
    np.save(Path(path_cov / f'{flux_idx}.npy'), MP)


def calc_ps_cov():

    n_ps = 81
    dl_ps_fg = np.load(f'./Dl_interp/pcfn/{n_ps}.npy')
    dl_ps_fg[dl_ps_fg<0] = 0
    print(f'{dl_ps_fg=}')
    print(np.any(dl_ps_fg < 0))
    cl_ps_fg = dl2cl(dl_ps_fg)

    cl_fg = gen_fg_cl()
    cl_cmb = gen_cmb_cl(beam=beam, lmax=lmax)

    cl_tot = cl_fg[:,:lmax+1] + cl_cmb


    plt.plot(cl_ps_fg)
    plt.plot(cl_tot[2])
    plt.show()

    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax)

    flux_idx = 0
    pix_ind = np.load(f'./pix_idx_qu/{flux_idx}.npy')
    obj_cov = CovCalculator(nside=nside, lmin=2, lmax=lmax, Cl_TT=cl_tot[0], Cl_EE=cl_tot[1], Cl_BB=cl_ps_fg*bl**2, Cl_TE=cl_tot[3], pixind=pix_ind, calc_opt='polarization', out_pol_opt='QU')
    MP = obj_cov.run_calc_cov()

    path_cov = Path('./cmb_qu_cov_ps')
    path_cov.mkdir(exist_ok=True, parents=True)
    # np.save(f'./test_class_cov/cmb.npy', MP)
    np.save(Path(path_cov / f'{flux_idx}.npy'), MP)


def do_th_fit():
    nstd = np.load(f'../../FGSim/NSTDNORTH/2048/{freq}.npy')
    df_mask = pd.read_csv(f'./mask/{freq}.csv')
    pcfn,_,_,_ = gen_map(rlz_idx=rlz_idx)

    obj = FitPolPS(m_q=pcfn[1], m_u=pcfn[2], freq=freq, nstd_q=nstd[1], nstd_u=nstd[2], flux_idx=0, df_mask=df_mask, df_ps=df_mask, lmax=lmax, nside=nside, radius_factor=1.5, beam=beam)

    # obj.calc_definite_fixed_cmb_cov()
    # obj.calc_covariance_matrix(mode='cmb+noise')
    num_ps, chi2dof, fit_P, fit_P_err, fit_phi, fit_phi_err = obj.fit_all(cov_mode='cmb+noise')
    path_res = Path('./parameter/th_have_all')
    path_res.mkdir(exist_ok=True, parents=True)
    np.save(path_res / Path(f'fit_P_{rlz_idx}.npy'), fit_P)
    np.save(path_res / Path(f'fit_phi_{rlz_idx}.npy'), fit_phi)

def do_cmb_fit():
    nstd = np.load(f'../../FGSim/NSTDNORTH/2048/{freq}.npy')
    df_mask = pd.read_csv(f'./mask/{freq}.csv')
    pcfn,_,_,_ = gen_map(rlz_idx=rlz_idx)

    obj = FitPolPS(m_q=pcfn[1], m_u=pcfn[2], freq=freq, nstd_q=nstd[1], nstd_u=nstd[2], flux_idx=0, df_mask=df_mask, df_ps=df_mask, lmax=lmax, nside=nside, radius_factor=1.5, beam=beam, cov_path='./cmb_qu_cov')

    # obj.calc_definite_fixed_cmb_cov()
    # obj.calc_covariance_matrix(mode='cmb+noise')
    num_ps, chi2dof, fit_P, fit_P_err, fit_phi, fit_phi_err = obj.fit_all(cov_mode='cmb+noise')
    path_res = Path('./parameter/cmb')
    path_res.mkdir(exist_ok=True, parents=True)
    np.save(path_res / Path(f'fit_P_{rlz_idx}.npy'), fit_P)
    np.save(path_res / Path(f'fit_phi_{rlz_idx}.npy'), fit_phi)


def do_ps_fit():
    nstd = np.load(f'../../FGSim/NSTDNORTH/2048/{freq}.npy')
    df_mask = pd.read_csv(f'./mask/{freq}.csv')
    pcfn,_,_,_ = gen_map(rlz_idx=rlz_idx)

    obj = FitPolPS(m_q=pcfn[1], m_u=pcfn[2], freq=freq, nstd_q=nstd[1], nstd_u=nstd[2], flux_idx=0, df_mask=df_mask, df_ps=df_mask, lmax=lmax, nside=nside, radius_factor=1.5, beam=beam, cov_path='./cmb_qu_cov_ps')

    obj.calc_definite_fixed_cmb_cov()
    obj.calc_covariance_matrix(mode='cmb+noise')
    num_ps, chi2dof, fit_P, fit_P_err, fit_phi, fit_phi_err = obj.fit_all(cov_mode='cmb+noise')
    path_res = Path('./parameter/ps')
    path_res.mkdir(exist_ok=True, parents=True)
    np.save(path_res / Path(f'fit_P_{rlz_idx}.npy'), fit_P)
    np.save(path_res / Path(f'fit_phi_{rlz_idx}.npy'), fit_phi)

# Part 4: check bias
def calc_e_cov():
    cl_fg = gen_fg_cl()
    cl_cmb = gen_cmb_cl(beam=beam, lmax=lmax)

    cl_tot = cl_fg + cl_cmb

    flux_idx = 0
    pix_ind = np.load(f'./pix_idx_qu/{flux_idx}.npy')
    obj_cov = CovCalculator(nside=nside, lmin=2, lmax=lmax, Cl_TT=cl_tot[0], Cl_EE=cl_tot[1], Cl_BB=np.zeros_like(cl_tot[1]), Cl_TE=cl_tot[3], pixind=pix_ind, calc_opt='polarization', out_pol_opt='QU')
    MP = obj_cov.run_calc_cov()

    path_cov = Path('./cmb_qu_cov')
    path_cov.mkdir(exist_ok=True, parents=True)
    # np.save(f'./test_class_cov/cmb.npy', MP)
    np.save(Path(path_cov / f'{flux_idx}.npy'), MP)

def do_e_fit():
    nstd = np.load(f'../../FGSim/NSTDNORTH/2048/{freq}.npy')
    df_mask = pd.read_csv(f'./mask/{freq}.csv')
    pcfn,_,_,_ = gen_map(rlz_idx=rlz_idx)

    obj = FitPolPS(m_q=pcfn[1], m_u=pcfn[2], freq=freq, nstd_q=nstd[1], nstd_u=nstd[2], flux_idx=0, df_mask=df_mask, df_ps=df_mask, lmax=lmax, nside=nside, radius_factor=1.5, beam=beam)

    obj.calc_definite_fixed_cmb_cov()
    obj.calc_covariance_matrix(mode='cmb+noise')
    num_ps, chi2dof, fit_P, fit_P_err, fit_phi, fit_phi_err = obj.fit_all(cov_mode='cmb+noise')
    path_res = Path('./parameter/th_e')
    path_res.mkdir(exist_ok=True, parents=True)
    np.save(path_res / Path(f'fit_P_{rlz_idx}.npy'), fit_P)
    np.save(path_res / Path(f'fit_phi_{rlz_idx}.npy'), fit_phi)

def calc_b_cov():
    cl_fg = gen_fg_cl()
    cl_cmb = gen_cmb_cl(beam=beam, lmax=lmax)

    cl_tot = cl_fg + cl_cmb

    flux_idx = 0
    pix_ind = np.load(f'./pix_idx_qu/{flux_idx}.npy')
    obj_cov = CovCalculator(nside=nside, lmin=2, lmax=lmax, Cl_TT=cl_tot[0], Cl_EE=np.zeros_like(cl_tot[0]), Cl_BB=cl_tot[2], Cl_TE=cl_tot[3], pixind=pix_ind, calc_opt='polarization', out_pol_opt='QU')
    MP = obj_cov.run_calc_cov()

    path_cov = Path('./cmb_qu_cov')
    path_cov.mkdir(exist_ok=True, parents=True)
    # np.save(f'./test_class_cov/cmb.npy', MP)
    np.save(Path(path_cov / f'{flux_idx}.npy'), MP)

def do_b_fit():
    nstd = np.load(f'../../FGSim/NSTDNORTH/2048/{freq}.npy')
    df_mask = pd.read_csv(f'./mask/{freq}.csv')
    pcfn,_,_,_ = gen_map(rlz_idx=rlz_idx)

    obj = FitPolPS(m_q=pcfn[1], m_u=pcfn[2], freq=freq, nstd_q=nstd[1], nstd_u=nstd[2], flux_idx=0, df_mask=df_mask, df_ps=df_mask, lmax=lmax, nside=nside, radius_factor=1.5, beam=beam)

    obj.calc_definite_fixed_cmb_cov()
    obj.calc_covariance_matrix(mode='cmb+noise')
    num_ps, chi2dof, fit_P, fit_P_err, fit_phi, fit_phi_err = obj.fit_all(cov_mode='cmb+noise')

    path_res = Path('./parameter/th_b')
    path_res.mkdir(exist_ok=True, parents=True)
    np.save(path_res / Path(f'fit_P_{rlz_idx}.npy'), fit_P)
    np.save(path_res / Path(f'fit_phi_{rlz_idx}.npy'), fit_phi)


# Part 5: Test estimate power spectrum
def test_cl():
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax)

    npix = hp.nside2npix(nside)
    nstd = np.load(f'../../FGSim/NSTDNORTH/2048/{freq}.npy')
    np.random.seed(seed=noise_seeds[rlz_idx])
    # noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))
    noise = nstd * np.random.normal(loc=0, scale=1, size=(3,npix))
    print(f"{np.std(noise[1])=}")

    cls = np.load('../../src/cmbsim/cmbdata/cmbcl_8k.npy')
    np.random.seed(seed=cmb_seeds[rlz_idx])
    # cmb_iqu = hp.synfast(cls.T, nside=nside, fwhm=np.deg2rad(beam)/60, new=True, lmax=1999)
    cmb_iqu = hp.synfast(cls.T, nside=nside, fwhm=np.deg2rad(beam)/60, new=True, lmax=lmax)

    cls_cn = hp.anafast(cmb_iqu+noise, lmax=lmax)[2]
    cls_n = hp.anafast(noise, lmax=lmax)[2]
    cls_c = hp.anafast(cmb_iqu, lmax=lmax)[2]
    np.save('./Dl_res/cls_cn.npy', cls_cn)
    np.save('./Dl_res/cls_n.npy', cls_n)
    np.save('./Dl_res/cls_c.npy', cls_c)

    cls_cn = np.load('./Dl_res/cls_cn.npy')
    cls_n = np.load('./Dl_res/cls_n.npy')
    cls_c = np.load('./Dl_res/cls_c.npy')

    l = np.arange(np.size(cls_cn))
    plt.loglog(l*(l+1)*cls_cn/bl**2/(2*np.pi), label='cmb + noise')
    plt.loglog(l*(l+1)*cls_c/bl**2/(2*np.pi), label='cmb')
    plt.loglog(l*(l+1)*cls_n/bl**2/(2*np.pi), label='noise')
    plt.loglog(l*(l+1)*(cls_cn-cls_n)/bl**2/(2*np.pi), label='debiased cmb + noise')
    plt.xlabel('$\\ell$')
    plt.ylabel('$D_\\ell^{BB}$')

    plt.legend()
    plt.show()

    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax, pol=True)[:,2]
    l_min_edges, l_max_edges = generate_bins(l_min_start=10, delta_l_min=10, l_max=lmax+1, fold=0.2)
    # delta_ell = 30
    # bin_dl = nmt.NmtBin.from_nside_linear(nside, nlb=delta_ell, is_Dell=True)
    # bin_dl = nmt.NmtBin.from_lmax_linear(lmax=lmax, nlb=30, is_Dell=True)
    bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)

    ell_arr = bin_dl.get_effective_ells()

    dl_cn = bin_dl.bin_cell(cls_cn/bl**2)
    dl_n = bin_dl.bin_cell(cls_n/bl**2)
    dl_c = bin_dl.bin_cell(cls_c/bl**2)

    dl_cn = bin_dl.bin_cell(cls_cn)
    dl_n = bin_dl.bin_cell(cls_n)
    dl_c = bin_dl.bin_cell(cls_c)


    # plt.loglog(ell_arr, dl_cn, label='binned cmb + noise')
    # plt.loglog(ell_arr, dl_n, label='binned noise')
    # plt.loglog(ell_arr, dl_c, label='binned cmb')

    # plt.loglog(ell_arr, dl_cn - dl_n, label='binned debiased cmb + noise')
    plt.loglog(ell_arr, (dl_cn - dl_n)/ell_arr, label='binned debiased cmb + noise / ell')
    # plt.loglog(ell_arr, (dl_cn - dl_n), label='debias')
    plt.xlabel('$\\ell$')
    # plt.ylabel('$\\ell \cdot C_\\ell$')
    plt.ylabel('$D_\\ell^{BB}$')

    plt.legend()
    plt.show()

def test_cfn():

    # npix = hp.nside2npix(nside)
    # l = np.arange(lmax+1)
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax, pol=True)[:,2]
    cl_cmb = np.load('../../src/cmbsim/cmbdata/cmbcl_8k.npy').T[2,:lmax+1]
    # cl_fg = gen_fg_cl()[2,:lmax+1] / bl**2


    # pcfn, cfn, cf, n = gen_map(rlz_idx=rlz_idx)
    # np.save('./pcfn.npy', pcfn)
    # np.save('./cfn.npy', cfn)
    # np.save('./cf.npy', cf)
    # np.save('./n.npy', n)

    # nstd = np.load(f'../../FGSim/NSTDNORTH/2048/{freq}.npy')
    # np.random.seed(seed=noise_seeds[rlz_idx])
    # # noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))
    # noise = nstd * np.random.normal(loc=0, scale=1, size=(3,npix))
    # print(f"{np.std(noise[1])=}")

    # cl_n = hp.anafast(noise, lmax=lmax)

    map_depth = 1.35
    nl = (map_depth/bl)**2 / 3437.728**2

    lmax_debias = np.argmax(nl[2:]/cl_cmb[2:]>1e5) + 2
    print(f'{lmax_debias=}')
    plt.loglog(nl/cl_cmb)
    plt.show()


    # plt.loglog(l*(l+1)*cl_cmb/(2*np.pi), label='cmb')
    # plt.loglog(l*(l+1)*cl_fg/(2*np.pi), label='fg')
    # plt.loglog(l*(l+1)*cl_n[2]/bl**2/(2*np.pi), label='n')
    # plt.loglog(l*(l+1)*nl/(2*np.pi), label='th n')
    # plt.legend()
    # plt.show()

def test_th_cov():
    lmax = 1300
    cl_fg = gen_fg_cl()
    cl_cmb = gen_cmb_cl(beam=beam, lmax=lmax)

    cl_tot = cl_fg + cl_cmb

    flux_idx = 0
    pix_ind = np.load(f'./pix_idx_qu/{flux_idx}.npy')
    path_cov = Path('./cov_test')
    path_cov.mkdir(exist_ok=True, parents=True)

    for lmax in [300,350,400,450]:
        obj_cov = CovCalculator(nside=nside, lmin=2, lmax=lmax, Cl_TT=cl_tot[0], Cl_EE=cl_tot[1], Cl_BB=cl_tot[2], Cl_TE=cl_tot[3], pixind=pix_ind, calc_opt='polarization', out_pol_opt='QU')
        MP = obj_cov.run_calc_cov()
        np.save(Path(path_cov / f'{lmax}.npy'), MP)

    # [500,550,600,650,700,1000,1300]:

def compute_relative_error(cov1: np.ndarray, cov2: np.ndarray, norm: str = "fro") -> float:
    """
    计算协方差矩阵的相对误差。

    :param cov1: 原始协方差矩阵 (numpy array)
    :param cov2: 比较协方差矩阵 (numpy array)
    :param norm: 使用的矩阵范数 ("fro" 或 "2")
    :return: 相对误差
    """
    if norm == "fro":
        diff_norm = np.linalg.norm(cov1 - cov2, ord="fro")
        base_norm = np.linalg.norm(cov1, ord="fro")
    elif norm == "2":
        diff_norm = np.linalg.norm(cov1 - cov2, ord=2)
        base_norm = np.linalg.norm(cov1, ord=2)
    else:
        raise ValueError("Unsupported norm type. Use 'fro' or '2'.")

    return diff_norm / base_norm

def check_cov_changes():
    cov_right = np.load('./cov_test/1300.npy')
    for lmax in [350, 400, 450, 500,550,600,650,700,1000,1300]:
        cov_1 = np.load(f'./cov_test/{lmax}.npy')
        # print(f'{cov_right=}')
        # print(f'{cov_1=}')
        # max_error = np.max(np.abs(cov_1-cov_right))
        # max_error = (cov_1[0,0] - cov_right[0,0]) / cov_right[0,0]
        max_error = compute_relative_error(cov_1, cov_right)
        print(f'{lmax=}, {max_error=}')


    # plt.figure(1)
    # cax = plt.imshow(np.abs(cov_right), cmap='viridis')
    # plt.title('th')
    # plt.colorbar(cax)
    # plt.show()

    # plt.figure(2)
    # cax = plt.imshow(np.abs(cov_right-cov_1), cmap='viridis')
    # plt.title('res')
    # plt.colorbar(cax)
    # plt.show()





if __name__=='__main__':
    # gen_mask()
    # check_mask()
    # estimate_powerspectrum()
    # estimate_powerspectrum_ori()
    # check_powerspectrum_ps_mask()
    # check_powerspectrum_ori_mask()
    # check_powerspectrum_one_rlz()
    # check_powerspectrum_rmse()
    # bandpower2powerspectrum()
    # check_input_cl()
    # calc_th_cov()
    # calc_cmb_cov()
    # calc_ps_cov()
    do_th_fit()
    # do_cmb_fit()
    # do_ps_fit()

    # calc_e_cov()
    # do_e_fit()

    # calc_b_cov()
    # do_b_fit()

    # test_cl()
    # test_cfn()
    # test_th_cov()

    # check_cov_changes()




