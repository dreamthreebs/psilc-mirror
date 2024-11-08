import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pandas as pd
import pymaster as nmt
import os,sys

from pathlib import Path
config_dir = Path(__file__).parent.parent
print(f'{config_dir=}')
sys.path.insert(0, str(config_dir))
from config import freq, lmax, nside, beam

l = np.arange(lmax+1)
nside = 2048
rlz_idx=0
threshold = 3

df = pd.read_csv('../../../FGSim/FreqBand')
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

    # cl_cmb = np.load('../../../src/cmbsim/cmbdata/cmbcl_8k.npy').T[2,:lmax+1]
    # plt.semilogy(l, l*(l+1)*cl_cmb / (2*np.pi), label='true cmb')

    # cl_fg = np.load('../../Cl_fg/data_1010/cl_fg.npy')[2,:lmax+1]
    # plt.semilogy(l, l*(l+1)*cl_fg/bl**2/(2*np.pi), label='input fg')
    # plt.semilogy(l, l*(l+1)*(cl_fg/bl**2+cl_cmb)/(2*np.pi), label='input cmb+fg')

    # cl_fg = np.load('../../Cl_fg/data_old/cl_fg_BB.npy')[:lmax+1]
    # plt.semilogy(l, l*(l+1)*(cl_fg/bl**2+cl_cmb)/(2*np.pi), label='input cmb+fg old')


    for rlz_idx in range(200):
        print(f'{rlz_idx=}')
        # nmt_22_n = np.load(f'./pcfn_dl/RMV/n/{rlz_idx}.npy')
        nmt_22_rmv_n = np.load(f'./pcfn_dl/RMV/rmv_n/{rlz_idx}.npy')
        nmt_22_apo_n = np.load(f'./pcfn_dl/RMV/apo_n/{rlz_idx}.npy')
        # nmt_15_n = np.load(f'../../benchmark_T_76/fit_res/pcfn_dl/RMV/n/{rlz_idx}.npy')

        # nmt_22_pcfn = np.load(f'./pcfn_dl/RMV/pcfn/{rlz_idx}.npy') - nmt_22_n
        # nmt_22_cfn = np.load(f'./pcfn_dl/RMV/cfn/{rlz_idx}.npy') - nmt_22_n
        # nmt_22_cf = np.load(f'./pcfn_dl/RMV/cf/{rlz_idx}.npy')

        nmt_22_rmv_pcfn = np.load(f'./pcfn_dl/RMV/rmv_pcfn/{rlz_idx}.npy') - nmt_22_rmv_n
        nmt_22_apo_pcfn = np.load(f'./pcfn_dl/RMV/apo/{rlz_idx}.npy') - nmt_22_apo_n

        # nmt_15_pcfn = np.load(f'../../benchmark_T_76/fit_res/pcfn_dl/RMV/pcfn/{rlz_idx}.npy') - nmt_15_n
        # nmt_15_cfn = np.load(f'../../benchmark_T_76/fit_res/pcfn_dl/RMV/cfn/{rlz_idx}.npy') - nmt_15_n
        # nmt_15_cf = np.load(f'../../benchmark_T_76/fit_res/pcfn_dl/RMV/cf/{rlz_idx}.npy')

        # nmt_15_pcfn_list.append(nmt_15_pcfn)
        # nmt_15_cfn_list.append(nmt_15_cfn)
        # nmt_15_cf_list.append(nmt_15_cf)
        # nmt_15_n_list.append(nmt_15_n)

        # nmt_22_pcfn_list.append(nmt_22_pcfn)
        # nmt_22_cfn_list.append(nmt_22_cfn)
        # nmt_22_cf_list.append(nmt_22_cf)
        # nmt_22_n_list.append(nmt_22_n)

        nmt_22_rmv_pcfn_list.append(nmt_22_rmv_pcfn)
        nmt_22_apo_pcfn_list.append(nmt_22_apo_pcfn)


    # nmt_15_pcfn_mean = np.mean(nmt_15_pcfn_list, axis=0)
    # nmt_15_cfn_mean = np.mean(nmt_15_cfn_list, axis=0)
    # nmt_15_cf_mean = np.mean(nmt_15_cf_list, axis=0)
    # nmt_15_n_mean = np.mean(nmt_15_n_list, axis=0)

    # nmt_22_pcfn_mean = np.mean(nmt_22_pcfn_list, axis=0)
    # nmt_22_cfn_mean = np.mean(nmt_22_cfn_list, axis=0)
    # nmt_22_cf_mean = np.mean(nmt_22_cf_list, axis=0)
    # nmt_22_n_mean = np.mean(nmt_22_n_list, axis=0)
    nmt_22_rmv_pcfn_mean = np.mean(nmt_22_rmv_pcfn_list, axis=0)
    nmt_22_apo_pcfn_mean = np.mean(nmt_22_apo_pcfn_list, axis=0)

    # plt.semilogy(ell_arr, nmt_15_pcfn_mean, label='nmt version 1.5 pcfn')
    # plt.semilogy(ell_arr, nmt_15_cfn_mean, label='nmt version 1.5 cfn')
    # plt.semilogy(ell_arr, nmt_15_cf_mean, label='nmt version 1.5 cf')
    # plt.semilogy(ell_arr, nmt_15_n_mean, label='nmt version 1.5 n')

    # plt.semilogy(ell_arr, nmt_22_pcfn_mean, label='nmt version 2.2 pcfn')
    # plt.semilogy(ell_arr, nmt_22_cfn_mean, label='nmt version 2.2 cfn')
    # plt.semilogy(ell_arr, nmt_22_cf_mean, label='nmt version 2.2 cf')
    # plt.semilogy(ell_arr, nmt_22_n_mean, label='nmt version 2.2 n')
    plt.semilogy(ell_arr, nmt_22_rmv_pcfn_mean, label='nmt version 2.2 rmv pcfn')
    plt.semilogy(ell_arr, nmt_22_apo_pcfn_mean, label='nmt version 2.2 apo pcfn')
    plt.legend()
    plt.show()

def check_inp():
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax, pol=True)[:,2]
    l_min_edges, l_max_edges = generate_bins(l_min_start=30, delta_l_min=30, l_max=lmax+1, fold=0.2)
    # delta_ell = 30
    # bin_dl = nmt.NmtBin.from_nside_linear(nside, nlb=delta_ell, is_Dell=True)
    # bin_dl = nmt.NmtBin.from_lmax_linear(lmax=lmax, nlb=30, is_Dell=True)
    bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    ell_arr = bin_dl.get_effective_ells()

    pcfn_list = []
    cfn_list = []
    inp_1_list = []
    inp_2_list = []
    inp_3_list = []
    inp_4_list = []

    for rlz_idx in range(200):
        n = np.load(f'./pcfn_dl/RMV/n/{rlz_idx}.npy')
        pcfn = np.load(f'./pcfn_dl/RMV/pcfn/{rlz_idx}.npy') - n
        cfn = np.load(f'./pcfn_dl/RMV/cfn/{rlz_idx}.npy') - n
        inp_1_n = np.load(f'pcfn_dl/INP_1108/inp_edge_m2_n/{rlz_idx}.npy')
        inp_1 = np.load(f'./pcfn_dl/INP_1108/inp_edge_m2_pcfn/{rlz_idx}.npy') - inp_1_n
        inp_2_n = np.load(f'./pcfn_dl/INP_1108/inp_edge_m3_n/{rlz_idx}.npy')
        inp_2 = np.load(f'./pcfn_dl/INP_1108/inp_edge_m3_pcfn/{rlz_idx}.npy') - inp_2_n
        inp_3_n = np.load(f'./pcfn_dl/INP_1108/inp_hole_m2_n/{rlz_idx}.npy')
        inp_3 = np.load(f'./pcfn_dl/INP_1108/inp_hole_m2_pcfn/{rlz_idx}.npy') - inp_3_n
        inp_4_n = np.load(f'./pcfn_dl/INP_1108/inp_hole_m3_n/{rlz_idx}.npy')
        inp_4 = np.load(f'./pcfn_dl/INP_1108/inp_hole_m3_pcfn/{rlz_idx}.npy') - inp_4_n

        pcfn_list.append(pcfn)
        cfn_list.append(cfn)
        inp_1_list.append(inp_1)
        inp_2_list.append(inp_2)
        inp_3_list.append(inp_3)
        inp_4_list.append(inp_4)

    pcfn_mean = np.mean(pcfn_list, axis=0)
    cfn_mean = np.mean(cfn_list, axis=0)
    inp_1_mean = np.mean(inp_1_list, axis=0)
    inp_2_mean = np.mean(inp_2_list, axis=0)
    inp_3_mean = np.mean(inp_3_list, axis=0)
    inp_4_mean = np.mean(inp_4_list, axis=0)

    pcfn_std = np.std(pcfn_list, axis=0)
    cfn_std = np.std(cfn_list, axis=0)
    inp_1_std = np.std(inp_1_list, axis=0)
    inp_2_std = np.std(inp_2_list, axis=0)
    inp_3_std = np.std(inp_3_list, axis=0)
    inp_4_std = np.std(inp_4_list, axis=0)



    plt.figure(1)
    plt.semilogy(ell_arr, pcfn_mean, label='point source + fg + cmb + noise')
    plt.semilogy(ell_arr, cfn_mean, label='fg + cmb + noise')
    plt.semilogy(ell_arr, inp_1_mean, label='inp m2 no hole')
    plt.semilogy(ell_arr, inp_2_mean, label='inp m3 no hole')
    plt.semilogy(ell_arr, inp_3_mean, label='inp m2 with hole')
    plt.semilogy(ell_arr, inp_4_mean, label='inp m3 with hole')
    plt.legend()
    plt.xlabel('$\\ell$')
    plt.ylabel('$D\\ell [\\mu K^2]$')
    plt.title('mean')

    plt.figure(2)
    plt.semilogy(ell_arr, pcfn_std, label='point source + fg + cmb + noise')
    plt.semilogy(ell_arr, cfn_std, label='fg + cmb + noise')
    plt.semilogy(ell_arr, inp_1_std, label='inp m2 no hole')
    plt.semilogy(ell_arr, inp_2_std, label='inp m3 no hole')
    plt.semilogy(ell_arr, inp_3_std, label='inp m2 with hole')
    plt.semilogy(ell_arr, inp_4_std, label='inp m3 with hole')
    plt.xlabel('$\\ell$')
    plt.ylabel('$\\Delta D\\ell [\\mu K^2]$')
    plt.title('standard deviation')

    plt.legend()
    plt.show()





# check()
check_inp()


