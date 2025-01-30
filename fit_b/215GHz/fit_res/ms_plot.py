import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
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

cf_list = []
cfn_list = []
pcfn_list = []

rmv_list = []
ps_mask_list = []
inp_list = []

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

# l_min_edges, l_max_edges = generate_bins(l_min_start=30, delta_l_min=30, l_max=lmax+1, fold=0.2)
# bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
bin_dl = nmt.NmtBin.from_lmax_linear(lmax=lmax, nlb=40, is_Dell=True)
ell_arr = bin_dl.get_effective_ells()
print(f'{ell_arr.shape=}')

def mean_and_std(sim_mode):
    for rlz_idx in range(1,200):
        print(f'{rlz_idx=}')

        n_qu = np.load(f'./pcfn_dl3/{sim_mode}/n/{rlz_idx}.npy')
        pcfn = np.load(f'./pcfn_dl3/{sim_mode}/pcfn/{rlz_idx}.npy') - n_qu
        cfn = np.load(f'./pcfn_dl3/{sim_mode}/cfn/{rlz_idx}.npy') - n_qu
        cf = np.load(f'./pcfn_dl3/{sim_mode}/cf/{rlz_idx}.npy')

        n_rmv = np.load(f'./pcfn_dl3/RMV/n/{rlz_idx}.npy')
        rmv_qu = np.load(f'./pcfn_dl3/RMV/{sim_mode}/{rlz_idx}.npy') - n_rmv

        n_ps_mask = np.load(f'./pcfn_dl3/PS_MASK/{sim_mode}/n/{rlz_idx}.npy')
        ps_mask = np.load(f'./pcfn_dl3/PS_MASK/{sim_mode}/pcfn/{rlz_idx}.npy') - n_ps_mask

        n_inp = np.load(f'./pcfn_dl3/INP/noise/{rlz_idx}.npy')
        inp = np.load(f'./pcfn_dl3/INP/{sim_mode}/{rlz_idx}.npy') - n_inp

        # plt.loglog(ell_arr, pcfn, label='pcfn')
        # plt.loglog(ell_arr, cfn, label='cfn')
        # plt.loglog(ell_arr, cf, label='cf')
        # plt.loglog(ell_arr, rmv_qu, label='rmv_qu')
        # plt.loglog(ell_arr, n_qu, label='n_qu')
        # plt.loglog(ell_arr, n_rmv, label='n_rmv')
        # plt.loglog(ell_arr, n_ps_mask, label='n_ps_mask')
        # plt.loglog(ell_arr, n_inp, label='n_inp')
        # plt.loglog(ell_arr, inp, label='inp')
        # plt.loglog(ell_arr, ps_mask, label='ps_mask')

        # plt.legend()
        # plt.show()

        cf_list.append(cf)
        cfn_list.append(cfn)
        pcfn_list.append(pcfn)

        rmv_list.append(rmv_qu)
        ps_mask_list.append(ps_mask)
        inp_list.append(inp)


    pcfn_mean = np.mean(pcfn_list, axis=0)
    cfn_mean = np.mean(cfn_list, axis=0)
    cf_mean = np.mean(cf_list, axis=0)

    rmv_mean = np.mean(rmv_list, axis=0)
    ps_mask_mean = np.mean(ps_mask_list, axis=0)
    inp_mean = np.mean(inp_list, axis=0)

    pcfn_std = np.std(pcfn_list, axis=0)
    cfn_std = np.std(cfn_list, axis=0)
    cf_std = np.std(cf_list, axis=0)

    rmv_std = np.std(rmv_list, axis=0)
    ps_mask_std = np.std(ps_mask_list, axis=0)
    inp_std = np.std(inp_list, axis=0)

    return pcfn_mean, cfn_mean, cf_mean, rmv_mean, ps_mask_mean, inp_mean, pcfn_std, cfn_std, cf_std, rmv_std, ps_mask_std, inp_std

# pcfn_mean, cfn_mean, cf_mean, rmv_mean, ps_mask_mean, inp_mean, _, _, _, _, _, _ = mean_and_std(sim_mode='MEAN')

# plt.figure(1)
# plt.scatter(ell_arr, pcfn_mean, label='pcfn', marker='.')
# plt.scatter(ell_arr, cfn_mean, label='cfn', marker='.')
# plt.scatter(ell_arr, cf_mean, label='cf', marker='.') 
# plt.scatter(ell_arr, rmv_mean, label='rmv', marker='.')
# plt.scatter(ell_arr, ps_mask_mean, label='ps_mask', marker='.')
# plt.scatter(ell_arr, inp_mean, label='inp', marker='.')
# plt.xlabel('$\\ell$')
# plt.ylabel('$D_\\ell^{BB} [\mu K^2]$')

# plt.loglog()
# plt.legend()
# plt.title('mean')

# _, _, _, _, _, _, pcfn_std, cfn_std, cf_std, rmv_std, ps_mask_std, inp_std = mean_and_std(sim_mode='STD')
pcfn_mean, cfn_mean, cf_mean, rmv_mean, ps_mask_mean, inp_mean, pcfn_std, cfn_std, cf_std, rmv_std, ps_mask_std, inp_std = mean_and_std(sim_mode='STD')

plt.figure(2)
plt.scatter(ell_arr, pcfn_std, label='pcfn', marker='.')
plt.scatter(ell_arr, cfn_std, label='cfn', marker='.')
plt.scatter(ell_arr, cf_std, label='cf', marker='.')
plt.scatter(ell_arr, rmv_std, label='rmv', marker='.')
plt.scatter(ell_arr, ps_mask_std, label='ps_mask', marker='.')
plt.scatter(ell_arr, inp_std, label='inp', marker='.')
plt.xlabel('$\\ell$')
plt.ylabel('$D_\\ell^{BB} [\mu K^2]$')

plt.loglog()
plt.legend()
plt.title('standard deviation')
plt.show()

lmax_eff = calc_lmax(beam=beam)
lmax_ell_arr = find_left_nearest_index_np(ell_arr, target=lmax_eff)
print(f'{ell_arr=}')
ell_arr = ell_arr[:lmax_ell_arr]
print(f'{ell_arr[:lmax_ell_arr]=}')
print(f'{lmax_ell_arr=}')
cl_cmb = np.load('/afs/ihep.ac.cn/users/w/wangyiming25/work/dc2/psilc/src/cmbsim/cmbdata/cmbcl_8k.npy').T
print(f'{cl_cmb.shape=}')
l = np.arange(lmax_eff+1)
cmb_binned = bin_dl.bin_cell(cls_in=cl_cmb[2, :lmax+1])

# Create figure with 2 subplots (main and subfigure), sharing the x-axis
fig, (ax_main, ax_sub) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

s = 5

# Set the y-axis to logarithmic scale for both the main plot and subfigure
ax_main.set_yscale('log')
ax_sub.set_yscale('log')

# Plot mean values in the main axis (no error bars here)
ax_main.scatter(ell_arr, pcfn_mean[:lmax_ell_arr], s=s, label='PS + CMB + FG + NOISE')
ax_main.scatter(ell_arr, cfn_mean[:lmax_ell_arr], s=s, label='CMB + FG + NOISE')
ax_main.scatter(ell_arr, rmv_mean[:lmax_ell_arr], s=s, label='Template Fitting method')
ax_main.scatter(ell_arr, ps_mask_mean[:lmax_ell_arr], s=s, label='Mask on QU')
ax_main.scatter(ell_arr, inp_mean[:lmax_ell_arr], s=s, label='Recycling + Inpaint on B')
# ax_main.plot(l, l*(l+1)*cl_cmb[2,:lmax_eff+1]/(2*np.pi), label='CMB input', color='black')
# ax_main.scatter(ell_arr, cmb_binned[:lmax_ell_arr], s=s, label='CMB input', color='black')

# Set labels and title for the main plot
ax_main.set_ylabel('$D_\\ell^{BB} [\mu K^2]$')
# ax_main.set_xlim(2, lmax_eff)
# ax_main.set_ylim(, lmax_eff)
ax_main.set_title('Debiased power spectra')
ax_main.legend()

# Plot standard deviation in the subfigure (using scatter with no error bars)
ax_sub.scatter(ell_arr, pcfn_std[:lmax_ell_arr], s=s, label='PS + CMB + FG + NOISE')
ax_sub.scatter(ell_arr, cfn_std[:lmax_ell_arr], s=s, label='CMB + FG + NOISE')
ax_sub.scatter(ell_arr, rmv_std[:lmax_ell_arr], s=s, label='Template Fitting method')
ax_sub.scatter(ell_arr, ps_mask_std[:lmax_ell_arr], s=s, label='Mask on QU')
ax_sub.scatter(ell_arr, inp_std[:lmax_ell_arr], s=s, label='Recycling + Inpaint on B')

# Set labels for the subfigure (only xlabel here)
ax_sub.set_xlabel('$\\ell$')
ax_sub.set_ylabel('Standard Deviation')
# ax_sub.set_ylim(5e-4, 3e-2)

# Adjust layout for better spacing
plt.tight_layout()
plt.subplots_adjust(hspace=0)

# path_fig = Path('/afs/ihep.ac.cn/users/w/wangyiming25/tmp/20250120')
# path_fig.mkdir(exist_ok=True, parents=True)
# plt.savefig(path_fig / Path(f'{freq}GHz.png'), dpi=300)

# Show plot
plt.show()

