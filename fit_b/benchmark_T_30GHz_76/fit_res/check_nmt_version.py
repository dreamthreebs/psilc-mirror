import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pandas as pd
import pymaster as nmt

from pathlib import Path

lmax = 500
l = np.arange(lmax+1)
nside = 2048
rlz_idx=0
threshold = 3

df = pd.read_csv('../../../FGSim/FreqBand')
freq = df.at[0, 'freq']
beam = df.at[0, 'beam']
print(f'{freq=}, {beam=}')

bin_mask = np.load('../../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5.npy')
apo_mask = np.load('../../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5APO_5.npy')
# ps_mask = np.load(f'../inpainting/mask/apo_ps_mask.npy')

noise_seeds = np.load('../../seeds_noise_2k.npy')
cmb_seeds = np.load('../../seeds_cmb_2k.npy')
fg_seeds = np.load('../../seeds_fg_2k.npy')

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

def check():
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax, pol=True)[:,2]
    l_min_edges, l_max_edges = generate_bins(l_min_start=30, delta_l_min=30, l_max=lmax+1, fold=0.2)
    # delta_ell = 30
    # bin_dl = nmt.NmtBin.from_nside_linear(nside, nlb=delta_ell, is_Dell=True)
    # bin_dl = nmt.NmtBin.from_lmax_linear(lmax=lmax, nlb=30, is_Dell=True)
    bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    ell_arr = bin_dl.get_effective_ells()

    nmt_15_pcfn_list = []
    nmt_22_pcfn_list = []
    nmt_22_rmv_pcfn_list = []
    nmt_22_rmv_n_list = []

    nmt_22_apo_pcfn_list = []
    # nmt_22_apo_n_list = []

    nmt_15_cfn_list = []
    nmt_22_cfn_list = []
    nmt_15_cf_list = []
    nmt_22_cf_list = []
    nmt_15_n_list = []
    nmt_22_n_list = []

    cl_cmb = np.load('../../../src/cmbsim/cmbdata/cmbcl_8k.npy').T[2,:lmax+1]
    plt.semilogy(l, l*(l+1)*cl_cmb / (2*np.pi), label='true cmb')

    cl_fg = np.load('../../Cl_fg/data_1010/cl_fg.npy')[2,:lmax+1]
    plt.semilogy(l, l*(l+1)*cl_fg/bl**2/(2*np.pi), label='input fg')
    plt.semilogy(l, l*(l+1)*(cl_fg/bl**2+cl_cmb)/(2*np.pi), label='input cmb+fg')

    cl_fg = np.load('../../Cl_fg/data_old/cl_fg_BB.npy')[:lmax+1]
    plt.semilogy(l, l*(l+1)*(cl_fg/bl**2+cl_cmb)/(2*np.pi), label='input cmb+fg old')


    for rlz_idx in range(200):
        print(f'{rlz_idx=}')
        nmt_22_n = np.load(f'./pcfn_dl/RMV/n/{rlz_idx}.npy')
        nmt_22_rmv_n = np.load(f'./pcfn_dl/RMV/rmv_n/{rlz_idx}.npy')
        nmt_22_apo_n = np.load(f'./pcfn_dl/RMV/apo_n/{rlz_idx}.npy')
        nmt_15_n = np.load(f'../../benchmark_T_76/fit_res/pcfn_dl/RMV/n/{rlz_idx}.npy')

        nmt_22_pcfn = np.load(f'./pcfn_dl/RMV/pcfn/{rlz_idx}.npy') - nmt_22_n
        nmt_22_cfn = np.load(f'./pcfn_dl/RMV/cfn/{rlz_idx}.npy') - nmt_22_n
        nmt_22_cf = np.load(f'./pcfn_dl/RMV/cf/{rlz_idx}.npy')
        nmt_22_rmv_pcfn = np.load(f'./pcfn_dl/RMV/rmv_pcfn/{rlz_idx}.npy') - nmt_22_rmv_n
        nmt_22_apo_pcfn = np.load(f'./pcfn_dl/RMV/apo/{rlz_idx}.npy') - nmt_22_apo_n

        nmt_15_pcfn = np.load(f'../../benchmark_T_76/fit_res/pcfn_dl/RMV/pcfn/{rlz_idx}.npy') - nmt_15_n
        nmt_15_cfn = np.load(f'../../benchmark_T_76/fit_res/pcfn_dl/RMV/cfn/{rlz_idx}.npy') - nmt_15_n
        nmt_15_cf = np.load(f'../../benchmark_T_76/fit_res/pcfn_dl/RMV/cf/{rlz_idx}.npy')

        nmt_15_pcfn_list.append(nmt_15_pcfn)
        nmt_15_cfn_list.append(nmt_15_cfn)
        nmt_15_cf_list.append(nmt_15_cf)
        nmt_15_n_list.append(nmt_15_n)

        nmt_22_pcfn_list.append(nmt_22_pcfn)
        nmt_22_cfn_list.append(nmt_22_cfn)
        nmt_22_cf_list.append(nmt_22_cf)
        nmt_22_n_list.append(nmt_22_n)
        nmt_22_rmv_pcfn_list.append(nmt_22_rmv_pcfn)
        nmt_22_apo_pcfn_list.append(nmt_22_apo_pcfn)


    nmt_15_pcfn_mean = np.mean(nmt_15_pcfn_list, axis=0)
    nmt_15_cfn_mean = np.mean(nmt_15_cfn_list, axis=0)
    nmt_15_cf_mean = np.mean(nmt_15_cf_list, axis=0)
    nmt_15_n_mean = np.mean(nmt_15_n_list, axis=0)

    nmt_22_pcfn_mean = np.mean(nmt_22_pcfn_list, axis=0)
    nmt_22_cfn_mean = np.mean(nmt_22_cfn_list, axis=0)
    nmt_22_cf_mean = np.mean(nmt_22_cf_list, axis=0)
    nmt_22_n_mean = np.mean(nmt_22_n_list, axis=0)
    nmt_22_rmv_pcfn_mean = np.mean(nmt_22_rmv_pcfn_list, axis=0)
    nmt_22_apo_pcfn_mean = np.mean(nmt_22_apo_pcfn_list, axis=0)

    # plt.semilogy(ell_arr, nmt_15_pcfn_mean, label='nmt version 1.5 pcfn')
    # plt.semilogy(ell_arr, nmt_15_cfn_mean, label='nmt version 1.5 cfn')
    # plt.semilogy(ell_arr, nmt_15_cf_mean, label='nmt version 1.5 cf')
    # plt.semilogy(ell_arr, nmt_15_n_mean, label='nmt version 1.5 n')

    plt.semilogy(ell_arr, nmt_22_pcfn_mean, label='nmt version 2.2 pcfn')
    # plt.semilogy(ell_arr, nmt_22_cfn_mean, label='nmt version 2.2 cfn')
    plt.semilogy(ell_arr, nmt_22_cf_mean, label='nmt version 2.2 cf')
    plt.semilogy(ell_arr, nmt_22_n_mean, label='nmt version 2.2 n')
    plt.semilogy(ell_arr, nmt_22_rmv_pcfn_mean, label='nmt version 2.2 rmv pcfn')
    plt.semilogy(ell_arr, nmt_22_apo_pcfn_mean, label='nmt version 2.2 apo pcfn')
    plt.legend()
    plt.show()

check()


