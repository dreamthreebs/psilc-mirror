import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
import pymaster as nmt
import glob
import os,sys

from pathlib import Path
config_dir = Path(__file__).parent.parent
print(f'{config_dir=}')
sys.path.insert(0, str(config_dir))
from config import freq, lmax, nside, beam

l = np.arange(lmax+1)

df = pd.read_csv('../../../FGSim/FreqBand')
print(f'{freq=}, {beam=}')

def calc_lmax(beam):
    lmax_eff = 2 * np.pi / np.deg2rad(beam) * 60
    print(f'{lmax_eff=}')
    return int(lmax_eff) + 1

def find_left_nearest_index_np(arr, target):
    # Find the indices of values less than or equal to the target
    valid_indices = np.where(arr <= target)[0]

    # If there are no valid indices, handle the case (e.g., return None)
    if valid_indices.size == 0:
        return None

    # Get the index of the largest value less than or equal to the target
    nearest_index = valid_indices[-1]  # The largest valid index
    return nearest_index + 1

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



def test_correlation():
    # l_min_edges, l_max_edges = generate_bins(l_min_start=30, delta_l_min=30, l_max=lmax+1, fold=0.2)
    l_min_edges, l_max_edges = generate_bins(l_min_start=42, delta_l_min=40, l_max=lmax+1, fold=0.1, l_threshold=400)
    bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    # bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    # bin_dl = nmt.NmtBin.from_lmax_linear(lmax=lmax, nlb=40)
    
    sim_mode = "STD"
    
    rlz_range = np.arange(1, 200)
    rlz_range_1k = np.arange(1, 5000)
    base_path = f'./pcfn_dl4/{sim_mode}'

    ell_arr = bin_dl.get_effective_ells()
    # cf_list = [np.load(f'{base_path}/cf/{rlz_idx}.npy') for rlz_idx in rlz_range]
    n_qu_list = [np.load(f'{base_path}/n/{rlz_idx}.npy') for rlz_idx in rlz_range]

    pcfn_list = [np.load(f'{base_path}/pcfn/{rlz_idx}.npy') - n_qu for rlz_idx, n_qu in zip(rlz_range, n_qu_list)]
    cfn_list = [np.load(f'{base_path}/cfn/{rlz_idx}.npy') - n_qu for rlz_idx, n_qu in zip(rlz_range, n_qu_list)]
    ps_cfn_cor_list = [np.load(f'./BIAS/test_correlation/ps_cfn_{rlz_idx}.npy') for rlz_idx in rlz_range]


    # print(f"{len(ps_mask_list)=}")
    pcfn_mean = np.mean(pcfn_list, axis=0)
    cfn_mean = np.mean(cfn_list, axis=0)
    ps_cfn_cor_mean = np.mean(ps_cfn_cor_list, axis=0)

    dl_ps = np.load(f'./BIAS/pcfn/0.npy')
    print(f'{ell_arr.shape=}')
    lmax_eff = calc_lmax(beam=beam)
    lmax_ell_arr = find_left_nearest_index_np(ell_arr, target=lmax_eff)
    print(f'{ell_arr=}')
    ell_arr = ell_arr[:lmax_ell_arr]
    print(f'{ell_arr[:lmax_ell_arr]=}')
    print(f'{lmax_ell_arr=}')
    cl_cmb = np.load('/afs/ihep.ac.cn/users/w/wangyiming25/work/dc2/psilc/src/cmbsim/cmbdata/cmbcl_8k.npy').T
    print(f'{cl_cmb.shape=}')
    l = np.arange(lmax_eff+1)
    dl_in = bin_dl.bin_cell(cl_cmb[2,:lmax+1])

    plt.figure(1)
    plt.plot(ell_arr, pcfn_mean[:lmax_ell_arr] - cfn_mean[:lmax_ell_arr], label='Cl with ps - Cl no ps', marker='.')
    plt.plot(ell_arr, pcfn_mean[:lmax_ell_arr] - cfn_mean[:lmax_ell_arr] - 2 * ps_cfn_cor_mean[:lmax_ell_arr], label='Cl with ps - Cl no ps - 2 * ps "noise" correlation', marker='.')
    plt.plot(ell_arr, dl_ps[:lmax_ell_arr], label='ps contribution from ps only map', marker='.', linestyle=':')

    plt.plot(ell_arr, dl_in[:lmax_ell_arr], label='CMB', marker='.', color='black')
    plt.loglog()
    plt.xlabel(r"$\ell$")
    plt.ylabel(r"$D_\ell^{BB}$")
    plt.title(f"{freq=}GHz")
    plt.legend()
    plt.show()

def test_rmv_correlation():
    # l_min_edges, l_max_edges = generate_bins(l_min_start=30, delta_l_min=30, l_max=lmax+1, fold=0.2)
    l_min_edges, l_max_edges = generate_bins(l_min_start=42, delta_l_min=40, l_max=lmax+1, fold=0.1, l_threshold=400)
    bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    # bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    # bin_dl = nmt.NmtBin.from_lmax_linear(lmax=lmax, nlb=40)
    
    sim_mode = "STD"
    
    rlz_range = np.arange(1, 200)
    rlz_range_1k = np.arange(1, 5000)
    base_path = f'./pcfn_dl4/{sim_mode}'

    ell_arr = bin_dl.get_effective_ells()
    # cf_list = [np.load(f'{base_path}/cf/{rlz_idx}.npy') for rlz_idx in rlz_range]
    n_qu_list = [np.load(f'{base_path}/n/{rlz_idx}.npy') for rlz_idx in rlz_range]

    # pcfn_list = [np.load(f'{base_path}/pcfn/{rlz_idx}.npy') - n_qu for rlz_idx, n_qu in zip(rlz_range, n_qu_list)]
    cfn_list = [np.load(f'{base_path}/cfn/{rlz_idx}.npy') for rlz_idx, n_qu in zip(rlz_range, n_qu_list)]

    rmv_list = [np.load(f'./pcfn_dl4/RMV/STD/{rlz_idx}.npy') for rlz_idx in rlz_range]
    rmv_cor_list = [np.load(f'./BIAS/rmv_cor/cor_{rlz_idx}.npy') for rlz_idx in rlz_range]

    rmv_bias_map_list = [np.load(f'./BIAS/rmv/bias_all_{rlz_idx}.npy') for rlz_idx in rlz_range]

    # print(f"{len(ps_mask_list)=}")
    # pcfn_mean = np.mean(pcfn_list, axis=0)
    cfn_mean = np.mean(cfn_list, axis=0)
    rmv_mean = np.mean(rmv_list, axis=0)
    rmv_cor_mean = np.mean(rmv_cor_list, axis=0)

    rmv_bias_map_mean = np.mean(rmv_bias_map_list, axis=0)

    # dl_ps = np.load(f'./BIAS/rmv/bias_all_0.npy')
    print(f'{ell_arr.shape=}')
    lmax_eff = calc_lmax(beam=beam)
    lmax_ell_arr = find_left_nearest_index_np(ell_arr, target=lmax_eff)
    print(f'{ell_arr=}')
    ell_arr = ell_arr[:lmax_ell_arr]
    print(f'{ell_arr[:lmax_ell_arr]=}')
    print(f'{lmax_ell_arr=}')
    cl_cmb = np.load('/afs/ihep.ac.cn/users/w/wangyiming25/work/dc2/psilc/src/cmbsim/cmbdata/cmbcl_8k.npy').T
    print(f'{cl_cmb.shape=}')
    l = np.arange(lmax_eff+1)
    dl_in = bin_dl.bin_cell(cl_cmb[2,:lmax+1])

    plt.figure(1)
    # plt.plot(ell_arr, rmv_mean[:lmax_ell_arr], label='rmv', marker='.')
    plt.plot(ell_arr, rmv_mean[:lmax_ell_arr] - 2 * rmv_cor_mean[:lmax_ell_arr] - cfn_mean[:lmax_ell_arr], label='Cl after rmv - Cl no ps - 2 * Cl ps after rmv * "noise" correlation', marker='.')
    plt.plot(ell_arr, rmv_bias_map_mean[:lmax_ell_arr], label='Cl from ps after removal map', marker='.', linestyle=':')

    plt.plot(ell_arr, dl_in[:lmax_ell_arr], label='CMB', marker='.', color='black')
    plt.loglog()
    plt.xlabel(r"$\ell$")
    plt.ylabel(r"$D_\ell^{BB}$")
    plt.title(f"{freq=}GHz")
    plt.legend()
    plt.show()

def test_inp_correlation():
    # l_min_edges, l_max_edges = generate_bins(l_min_start=30, delta_l_min=30, l_max=lmax+1, fold=0.2)
    l_min_edges, l_max_edges = generate_bins(l_min_start=42, delta_l_min=40, l_max=lmax+1, fold=0.1, l_threshold=400)
    bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    # bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    # bin_dl = nmt.NmtBin.from_lmax_linear(lmax=lmax, nlb=40)
    
    sim_mode = "STD"
    
    rlz_range = np.arange(1, 200)
    rlz_range_1k = np.arange(1, 5000)
    base_path = f'./pcfn_dl4/{sim_mode}'

    ell_arr = bin_dl.get_effective_ells()
    # cf_list = [np.load(f'{base_path}/cf/{rlz_idx}.npy') for rlz_idx in rlz_range]
    n_qu_list = [np.load(f'{base_path}/n/{rlz_idx}.npy') for rlz_idx in rlz_range]

    # pcfn_list = [np.load(f'{base_path}/pcfn/{rlz_idx}.npy') - n_qu for rlz_idx, n_qu in zip(rlz_range, n_qu_list)]
    cfn_list = [np.load(f'{base_path}/cfn/{rlz_idx}.npy') for rlz_idx, n_qu in zip(rlz_range, n_qu_list)]
    inp_list = [np.load(f'./pcfn_dl4/INP/STD/{rlz_idx}.npy') for rlz_idx in rlz_range]
    cfn_b_list = [np.load(f'./BIAS/inp/cfn_{rlz_idx}.npy') for rlz_idx in rlz_range]
    cfn_out_b_list = [np.load(f'./BIAS/inp/inp_cfn_{rlz_idx}.npy') for rlz_idx in rlz_range]



    # print(f"{len(ps_mask_list)=}")
    # pcfn_mean = np.mean(pcfn_list, axis=0)
    cfn_mean = np.mean(cfn_list, axis=0)
    inp_mean = np.mean(inp_list, axis=0)
    cfn_b_mean = np.mean(cfn_b_list, axis=0)
    cfn_out_b_mean = np.mean(cfn_out_b_list, axis=0)


    # dl_ps = np.load(f'./BIAS/rmv/bias_all_0.npy')
    dl_unresolved_ps = np.load(f'./BIAS/unresolved_ps/0.npy')
    print(f'{ell_arr.shape=}')
    lmax_eff = calc_lmax(beam=beam)
    lmax_ell_arr = find_left_nearest_index_np(ell_arr, target=lmax_eff)
    print(f'{ell_arr=}')
    ell_arr = ell_arr[:lmax_ell_arr]
    print(f'{ell_arr[:lmax_ell_arr]=}')
    print(f'{lmax_ell_arr=}')
    cl_cmb = np.load('/afs/ihep.ac.cn/users/w/wangyiming25/work/dc2/psilc/src/cmbsim/cmbdata/cmbcl_8k.npy').T
    print(f'{cl_cmb.shape=}')
    l = np.arange(lmax_eff+1)
    dl_in = bin_dl.bin_cell(cl_cmb[2,:lmax+1])

    plt.figure(1)
    # plt.plot(ell_arr, rmv_mean[:lmax_ell_arr], label='rmv', marker='.')
    plt.plot(ell_arr, inp_mean[:lmax_ell_arr] - cfn_mean[:lmax_ell_arr], label='Cl inp - Cl cfn', marker='.')
    plt.plot(ell_arr, inp_mean[:lmax_ell_arr] - cfn_b_mean[:lmax_ell_arr], label='Cl inp - Cl cfn from B map', marker='.')
    plt.plot(ell_arr, dl_unresolved_ps[:lmax_ell_arr], label='unresolved point sources', marker='.')
    plt.plot(ell_arr, cfn_mean[:lmax_ell_arr], label='cfn from qu', marker='.')
    plt.plot(ell_arr, cfn_b_mean[:lmax_ell_arr], label='cfn from b', marker='.')
    plt.plot(ell_arr, cfn_out_b_mean[:lmax_ell_arr], label='cfn after inp from b', marker='.')
    plt.plot(ell_arr, cfn_out_b_mean[:lmax_ell_arr] - cfn_b_mean[:lmax_ell_arr], label='residual cfn inp - input', marker='.')
    plt.plot(ell_arr, np.abs(cfn_out_b_mean[:lmax_ell_arr] - cfn_b_mean[:lmax_ell_arr]), label='absolute residual cfn inp - input', marker='.')

    plt.plot(ell_arr, dl_in[:lmax_ell_arr], label='CMB', marker='.', color='black')

    plt.loglog()
    plt.xlabel(r"$\ell$")
    plt.ylabel(r"$D_\ell^{BB}$")
    plt.title(f"{freq=}GHz")
    plt.legend()
    plt.show()


    plt.figure(2)
    plt.plot(ell_arr, (inp_mean[:lmax_ell_arr] - cfn_mean[:lmax_ell_arr])/dl_in[:lmax_ell_arr], label='Cl inp - Cl cfn', marker='.')
    plt.plot(ell_arr, (inp_mean[:lmax_ell_arr] - cfn_b_mean[:lmax_ell_arr])/dl_in[:lmax_ell_arr], label='Cl inp - Cl cfn from B map', marker='.')
    plt.plot(ell_arr, dl_unresolved_ps[:lmax_ell_arr]/dl_in[:lmax_ell_arr], label='unresolved point sources', marker='.')
    # plt.plot(ell_arr, cfn_mean[:lmax_ell_arr]/dl_in[:lmax_ell_arr], label='cfn from qu', marker='.')
    # plt.plot(ell_arr, cfn_b_mean[:lmax_ell_arr]/dl_in[:lmax_ell_arr], label='cfn from b', marker='.')

    # plt.loglog()
    plt.xlabel(r"$\ell$")
    plt.ylabel(r"relateive $D_\ell^{BB}$")
    plt.title(f"{freq=}GHz")
    plt.legend()
    plt.show()

def test_inp():
    # l_min_edges, l_max_edges = generate_bins(l_min_start=30, delta_l_min=30, l_max=lmax+1, fold=0.2)
    l_min_edges, l_max_edges = generate_bins(l_min_start=42, delta_l_min=40, l_max=lmax+1, fold=0.1, l_threshold=400)
    bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    # bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    # bin_dl = nmt.NmtBin.from_lmax_linear(lmax=lmax, nlb=40)
    
    sim_mode = "STD"
    
    rlz_range = np.arange(1, 200)
    rlz_range_1k = np.arange(1, 5000)
    base_path = f'./pcfn_dl4/{sim_mode}'

    ell_arr = bin_dl.get_effective_ells()
    # cf_list = [np.load(f'{base_path}/cf/{rlz_idx}.npy') for rlz_idx in rlz_range]
    n_qu_list = [np.load(f'{base_path}/n/{rlz_idx}.npy') for rlz_idx in rlz_range]

    # pcfn_list = [np.load(f'{base_path}/pcfn/{rlz_idx}.npy') - n_qu for rlz_idx, n_qu in zip(rlz_range, n_qu_list)]
    inp_list = [np.load(f'./BIAS/inp_bias/inp_{rlz_idx}.npy') for rlz_idx in rlz_range]
    cfn_list = [np.load(f'./BIAS/inp_bias/cfn_{rlz_idx}.npy') for rlz_idx in rlz_range]



    # print(f"{len(ps_mask_list)=}")
    # pcfn_mean = np.mean(pcfn_list, axis=0)
    cfn_mean = np.mean(cfn_list, axis=0)
    inp_mean = np.mean(inp_list, axis=0)
    # cfn_b_mean = np.mean(cfn_b_list, axis=0)
    # cfn_out_b_mean = np.mean(cfn_out_b_list, axis=0)


    # dl_ps = np.load(f'./BIAS/rmv/bias_all_0.npy')
    dl_unresolved_ps = np.load(f'./BIAS/unresolved_ps/0.npy')
    dl_unresolved_ps_inp = np.load(f'./BIAS/inp_bias/unresolved_ps_0.npy')
    print(f'{ell_arr.shape=}')
    lmax_eff = calc_lmax(beam=beam)
    lmax_ell_arr = find_left_nearest_index_np(ell_arr, target=lmax_eff)
    print(f'{ell_arr=}')
    ell_arr = ell_arr[:lmax_ell_arr]
    print(f'{ell_arr[:lmax_ell_arr]=}')
    print(f'{lmax_ell_arr=}')
    cl_cmb = np.load('/afs/ihep.ac.cn/users/w/wangyiming25/work/dc2/psilc/src/cmbsim/cmbdata/cmbcl_8k.npy').T
    print(f'{cl_cmb.shape=}')
    l = np.arange(lmax_eff+1)
    dl_in = bin_dl.bin_cell(cl_cmb[2,:lmax+1])

    plt.figure(1)
    # plt.plot(ell_arr, rmv_mean[:lmax_ell_arr], label='rmv', marker='.')
    plt.plot(ell_arr, inp_mean[:lmax_ell_arr] - cfn_mean[:lmax_ell_arr], label='Cl inp - Cl cfn from B map', marker='.')
    plt.plot(ell_arr, np.abs(inp_mean[:lmax_ell_arr] - cfn_mean[:lmax_ell_arr]), label='abs Cl inp - Cl cfn from B map', marker='.')
    plt.plot(ell_arr, dl_unresolved_ps[:lmax_ell_arr], label='unresolved point sources', marker='.')
    plt.plot(ell_arr, dl_unresolved_ps_inp[:lmax_ell_arr], label='unresolved point sources in inpainting', marker='.')
    plt.plot(ell_arr, cfn_mean[:lmax_ell_arr], label='cfn', marker='.')
    plt.plot(ell_arr, inp_mean[:lmax_ell_arr], label='inp', marker='.')

    plt.plot(ell_arr, dl_in[:lmax_ell_arr], label='CMB', marker='.', color='black')

    plt.loglog()
    plt.xlabel(r"$\ell$")
    plt.ylabel(r"$D_\ell^{BB}$")
    plt.title(f"{freq=}GHz")
    plt.legend()
    plt.show()

def test_mask():
    # l_min_edges, l_max_edges = generate_bins(l_min_start=30, delta_l_min=30, l_max=lmax+1, fold=0.2)
    l_min_edges, l_max_edges = generate_bins(l_min_start=42, delta_l_min=40, l_max=lmax+1, fold=0.1, l_threshold=400)
    bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    # bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    # bin_dl = nmt.NmtBin.from_lmax_linear(lmax=lmax, nlb=40)
    
    sim_mode = "STD"
    
    rlz_range = np.arange(1, 200)
    rlz_range_1k = np.arange(1, 5000)
    base_path = f'./pcfn_dl4/{sim_mode}'

    ell_arr = bin_dl.get_effective_ells()
    # cf_list = [np.load(f'{base_path}/cf/{rlz_idx}.npy') for rlz_idx in rlz_range]
    n_qu_list = [np.load(f'{base_path}/n/{rlz_idx}.npy') for rlz_idx in rlz_range]




    dl_unresolved_ps = np.load(f'./BIAS/unresolved_ps/0.npy')
    dl_unresolved_ps_masking = np.load(f'./BIAS/mask/bias_all_0.npy')
    dl_unresolved_ps_no_eblkg = np.load(f'./BIAS/mask/no_eblkg_0.npy')
    dl_unresolved_ps_inp = np.load(f'./BIAS/inp_bias/unresolved_ps_0.npy')
    dl_unresolved_ps_ps = np.load(f'./BIAS/mask/no_eblkg_ps_0.npy')
    dl_unresolved_ps_bin_mask = np.load(f'./BIAS/mask/no_eblkg_bin_mask_0.npy')
    dl_unresolved_ps_no_apo = np.load(f'./BIAS/mask/no_eblkg_no_apo_0.npy')
    dl_unresolved_ps_b = np.load(f'./BIAS/mask/mask_on_b_0.npy')
    dl_unresolved_ps_b_2deg = np.load(f'./BIAS/mask/mask_C1_3_0.npy')
    dl_unresolved_ps_3nside = np.load(f'./BIAS/mask/3nside_2deg_0.npy')
    dl_mask = np.load(f'./BIAS/mask/mask_itself_0.npy')
    dl_mask_apo = np.load(f'./BIAS/mask/apo_mask_itself_0.npy')
    print(f'{ell_arr.shape=}')
    lmax_eff = calc_lmax(beam=beam)
    lmax_ell_arr = find_left_nearest_index_np(ell_arr, target=lmax_eff)
    print(f'{ell_arr=}')
    ell_arr = ell_arr[:lmax_ell_arr]
    print(f'{ell_arr[:lmax_ell_arr]=}')
    print(f'{lmax_ell_arr=}')
    cl_cmb = np.load('/afs/ihep.ac.cn/users/w/wangyiming25/work/dc2/psilc/src/cmbsim/cmbdata/cmbcl_8k.npy').T
    print(f'{cl_cmb.shape=}')
    l = np.arange(lmax_eff+1)
    dl_in = bin_dl.bin_cell(cl_cmb[2,:lmax+1])

    plt.figure(1)
    # plt.plot(ell_arr, rmv_mean[:lmax_ell_arr], label='rmv', marker='.')
    plt.plot(ell_arr, dl_unresolved_ps[:lmax_ell_arr], label='unresolved point sources', marker='.')
    plt.plot(ell_arr, dl_unresolved_ps_masking[:lmax_ell_arr], label='unresolved point sources in ps mask', marker='.')
    plt.plot(ell_arr, dl_unresolved_ps_no_eblkg[:lmax_ell_arr], label='unresolved point sources in no eblkg', marker='.')
    plt.plot(ell_arr, dl_unresolved_ps_inp[:lmax_ell_arr], label='unresolved point sources inpainting', marker='.')
    plt.plot(ell_arr, dl_unresolved_ps_b[:lmax_ell_arr], label='unresolved point sources on B C1 1deg', marker='.')
    plt.plot(ell_arr, dl_unresolved_ps_b_2deg[:lmax_ell_arr], label='unresolved point sources on B C1 2deg', marker='.')
    plt.plot(ell_arr, dl_mask[:lmax_ell_arr], label='mask', marker='.')
    plt.plot(ell_arr, dl_mask_apo[:lmax_ell_arr], label='mask apo', marker='.')
    plt.plot(ell_arr, dl_unresolved_ps_3nside[:lmax_ell_arr], label='mask 3nside', marker='.')
    # plt.plot(ell_arr, dl_unresolved_ps_ps[:lmax_ell_arr], label='unresolved ps ps', marker='.')
    # plt.plot(ell_arr, dl_unresolved_ps_no_apo[:lmax_ell_arr], label='unresolved ps no apo', marker='.')

    plt.plot(ell_arr, dl_in[:lmax_ell_arr], label='CMB', marker='.', color='black')

    plt.loglog()
    plt.xlabel(r"$\ell$")
    plt.ylabel(r"$D_\ell^{BB}$")
    plt.title(f"{freq=}GHz")
    plt.legend()
    plt.show()

def test_mask_cmb():
    # l_min_edges, l_max_edges = generate_bins(l_min_start=30, delta_l_min=30, l_max=lmax+1, fold=0.2)
    l_min_edges, l_max_edges = generate_bins(l_min_start=42, delta_l_min=40, l_max=lmax+1, fold=0.1, l_threshold=400)
    bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    # bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    # bin_dl = nmt.NmtBin.from_lmax_linear(lmax=lmax, nlb=40)
    
    sim_mode = "STD"
    
    rlz_range = np.arange(1, 200)
    rlz_range_1k = np.arange(1, 5000)
    base_path = f'./pcfn_dl4/{sim_mode}'

    ell_arr = bin_dl.get_effective_ells()
    # cf_list = [np.load(f'{base_path}/cf/{rlz_idx}.npy') for rlz_idx in rlz_range]
    n_qu_list = [np.load(f'{base_path}/n/{rlz_idx}.npy') for rlz_idx in rlz_range]

    c_list = [np.load(f"./BIAS/test_mask/c_{rlz_idx}.npy") for rlz_idx in rlz_range]
    c_apo_list = [np.load(f"./BIAS/test_mask/c_apo_{rlz_idx}.npy") for rlz_idx in rlz_range]
    full_c_list = [np.load(f"./BIAS/test_mask/full_c_{rlz_idx}.npy") for rlz_idx in rlz_range]
    c_0 = np.load(f"./BIAS/test_mask/c_0.npy")
    c_th_apo = np.load(f"./BIAS/test_mask/th_c_apo_0.npy")
    c_th_ps = np.load(f"./BIAS/test_mask/th_c_ps_0.npy")


    c_mean = np.mean(c_list, axis=0)
    c_std = np.std(c_list, axis=0) / np.sqrt(np.size(rlz_range))
    c_apo_mean = np.mean(c_apo_list, axis=0)
    c_apo_std = np.std(c_apo_list, axis=0) / np.sqrt(np.size(rlz_range))
    full_c_mean = np.mean(full_c_list, axis=0)

    # dl_unresolved_ps = np.load(f'./BIAS/unresolved_ps/0.npy')
    # dl_unresolved_ps_masking = np.load(f'./BIAS/mask/bias_all_0.npy')
    # dl_unresolved_ps_no_eblkg = np.load(f'./BIAS/mask/no_eblkg_0.npy')
    # dl_unresolved_ps_inp = np.load(f'./BIAS/inp_bias/unresolved_ps_0.npy')
    # dl_unresolved_ps_ps = np.load(f'./BIAS/mask/no_eblkg_ps_0.npy')
    # dl_unresolved_ps_bin_mask = np.load(f'./BIAS/mask/no_eblkg_bin_mask_0.npy')
    # dl_unresolved_ps_no_apo = np.load(f'./BIAS/mask/no_eblkg_no_apo_0.npy')
    # dl_unresolved_ps_b = np.load(f'./BIAS/mask/mask_on_b_0.npy')
    # dl_unresolved_ps_b_2deg = np.load(f'./BIAS/mask/mask_C1_3_0.npy')

    # dl_mask = np.load(f'./BIAS/mask/mask_itself_0.npy')
    # dl_mask_apo = np.load(f'./BIAS/mask/apo_mask_itself_0.npy')

    print(f'{ell_arr.shape=}')
    lmax_eff = calc_lmax(beam=beam)
    lmax_ell_arr = find_left_nearest_index_np(ell_arr, target=lmax_eff)
    print(f'{ell_arr=}')
    ell_arr = ell_arr[:lmax_ell_arr]
    print(f'{ell_arr[:lmax_ell_arr]=}')
    print(f'{lmax_ell_arr=}')
    cl_cmb = np.load('/afs/ihep.ac.cn/users/w/wangyiming25/work/dc2/psilc/src/cmbsim/cmbdata/cmbcl_8k.npy').T
    print(f'{cl_cmb.shape=}')
    l = np.arange(lmax_eff+1)
    dl_in = bin_dl.bin_cell(cl_cmb[2,:lmax+1])

    plt.figure(1)
    # plt.plot(ell_arr, rmv_mean[:lmax_ell_arr], label='rmv', marker='.')
    plt.plot(ell_arr, c_mean[:lmax_ell_arr], label='c mean ps mask', marker='.')
    plt.plot(ell_arr, c_apo_mean[:lmax_ell_arr], label='c mean apo edge', marker='.')
    plt.plot(ell_arr, full_c_mean[:lmax_ell_arr], label='c mean full', marker='.')
    plt.plot(ell_arr, c_th_apo[:lmax_ell_arr], label='c th apo', marker='.', linestyle='--')
    plt.plot(ell_arr, c_th_ps[:lmax_ell_arr], label='c th ps', marker='.', linestyle='--')

    # plt.plot(ell_arr, c_0[:lmax_ell_arr], label='c 0', marker='.', linestyle=':')
    plt.plot(ell_arr, dl_in[:lmax_ell_arr], label='CMB', marker='.', color='black')

    plt.loglog()
    plt.xlabel(r"$\ell$")
    plt.ylabel(r"$D_\ell^{BB}$")
    plt.title(f"{freq=}GHz")
    plt.legend()
    plt.show()

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_axes((.1,.3,.8,.6))
    ax.plot(ell_arr, c_th_apo[:lmax_ell_arr], 'b-', label='Input, bpw-convolved')
    ax.plot(ell_arr, dl_in[:lmax_ell_arr], 'r--', label='Input, naive binning')
    ax.errorbar(ell_arr, c_apo_mean[:lmax_ell_arr], yerr=c_apo_std[:lmax_ell_arr], fmt='k.', label=r'$C_\ell$ from sims')
    ax.set_yscale('log')
    ax.set_ylabel(r'$D_\ell$', fontsize=15)
    ax.set_xticklabels([])
    ax.legend()
    ax.set_xlim(0, lmax_eff)
    # ax.axvspan(2*nside, 3*nside, facecolor='#AAAAAA', alpha=0.5)

    ax = fig.add_axes((.1,.1,.8,.2))
    ax.errorbar(ell_arr, (c_apo_mean[:lmax_ell_arr]-dl_in[:lmax_ell_arr])/c_apo_std[:lmax_ell_arr], yerr=np.ones_like(ell_arr), fmt='r.--')
    ax.errorbar(ell_arr, (c_apo_mean[:lmax_ell_arr]-c_th_apo[:lmax_ell_arr])/c_apo_std[:lmax_ell_arr], yerr=np.ones_like(ell_arr), fmt='b.')
    ax.axhline(0, c='k', ls='--')
    ax.set_ylim([-2.9, 2.9])
    ax.set_xlim(0,lmax_eff)
    # ax.axvspan(2*nside, 3*nside, facecolor='#AAAAAA', alpha=0.5)
    ax.set_ylabel(r'$\Delta D_\ell/\sigma(D_\ell)$', fontsize=16)
    ax.set_xlabel(r'$\ell$', fontsize=16);
    plt.show()

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_axes((.1,.3,.8,.6))
    ax.plot(ell_arr, c_th_ps[:lmax_ell_arr], 'b-', label='Input, bpw-convolved')
    ax.plot(ell_arr, dl_in[:lmax_ell_arr], 'r--', label='Input, naive binning')
    ax.errorbar(ell_arr, c_mean[:lmax_ell_arr], yerr=c_std[:lmax_ell_arr], fmt='k.', label=r'$C_\ell$ from sims')
    ax.set_yscale('log')
    ax.set_ylabel(r'$D_\ell$', fontsize=15)
    ax.set_xticklabels([])
    ax.legend()
    ax.set_xlim(0, lmax_eff)
    # ax.axvspan(2*nside, 3*nside, facecolor='#AAAAAA', alpha=0.5)

    ax = fig.add_axes((.1,.1,.8,.2))
    ax.errorbar(ell_arr, (c_mean[:lmax_ell_arr]-dl_in[:lmax_ell_arr])/c_std[:lmax_ell_arr], yerr=np.ones_like(ell_arr), fmt='r.--')
    ax.errorbar(ell_arr, (c_mean[:lmax_ell_arr]-c_th_ps[:lmax_ell_arr])/c_std[:lmax_ell_arr], yerr=np.ones_like(ell_arr), fmt='b.')
    ax.axhline(0, c='k', ls='--')
    ax.set_ylim([-2.9, 2.9])
    ax.set_xlim(0,lmax_eff)
    # ax.axvspan(2*nside, 3*nside, facecolor='#AAAAAA', alpha=0.5)
    ax.set_ylabel(r'$\Delta D_\ell/\sigma(D_\ell)$', fontsize=16)
    ax.set_xlabel(r'$\ell$', fontsize=16);
    plt.show()
 
    # plt.figure(2)
    # plt.plot(ell_arr, np.abs(c_apo_mean[:lmax_ell_arr]-c_mean[:lmax_ell_arr])/c_mean[:lmax_ell_arr], label='relative difference', marker='.')
    # # plt.loglog()
    # plt.xlabel(r"$\ell$")
    # plt.ylabel(r"$difference D_\ell^{BB}$")
    # plt.title(f"{freq=}GHz")
    # plt.legend()
    # plt.show()



# test_correlation()
# test_rmv_correlation()
# test_inp_correlation()
# test_inp()
# test_mask()
test_mask_cmb()

