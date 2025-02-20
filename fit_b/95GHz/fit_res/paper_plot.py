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


# l_min_edges, l_max_edges = generate_bins(l_min_start=30, delta_l_min=30, l_max=lmax+1, fold=0.2)
l_min_edges, l_max_edges = generate_bins(l_min_start=42, delta_l_min=40, l_max=lmax+1, fold=0.1, l_threshold=400)
bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
# bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
# bin_dl = nmt.NmtBin.from_lmax_linear(lmax=lmax, nlb=40)
ell_arr = bin_dl.get_effective_ells()
print(f'{ell_arr.shape=}')

def mean_and_std(sim_mode):
    for rlz_idx in range(1,200):
        print(f'{rlz_idx=}')

        n_qu = np.load(f'./pcfn_dl4/{sim_mode}/n/{rlz_idx}.npy')
        pcfn = np.load(f'./pcfn_dl4/{sim_mode}/pcfn/{rlz_idx}.npy') - n_qu
        cfn = np.load(f'./pcfn_dl4/{sim_mode}/cfn/{rlz_idx}.npy') - n_qu
        cf = np.load(f'./pcfn_dl4/{sim_mode}/cf/{rlz_idx}.npy')

        n_rmv = np.load(f'./pcfn_dl4/RMV/n/{rlz_idx}.npy')
        rmv_qu = np.load(f'./pcfn_dl4/RMV/{sim_mode}/{rlz_idx}.npy') - n_rmv

        n_ps_mask = np.load(f'./pcfn_dl4/PS_MASK/{sim_mode}/n/{rlz_idx}.npy')
        ps_mask = np.load(f'./pcfn_dl4/PS_MASK/{sim_mode}/pcfn/{rlz_idx}.npy') - n_ps_mask

        n_inp = np.load(f'./pcfn_dl4/INP/noise/{rlz_idx}.npy')
        inp = np.load(f'./pcfn_dl4/INP/{sim_mode}/{rlz_idx}.npy') - n_inp

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

print(f'{ell_arr.shape=}')
print(f'{pcfn_std.shape=}')

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
dl_in = bin_dl.bin_cell(cl_cmb[2,:lmax+1])

# Create figure with 2 subplots (main and subfigure), sharing the x-axis
fig, (ax_main, ax_sub) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

s = 5

# Set the y-axis to logarithmic scale for both the main plot and subfigure
ax_main.set_yscale('log')
ax_sub.set_yscale('log')
ax_main.set_xscale('log')
ax_sub.set_xscale('log')


# Plot mean values in the main axis (no error bars here)
ax_main.errorbar(ell_arr*0.985, pcfn_mean[:lmax_ell_arr], yerr=pcfn_std[:lmax_ell_arr], label='PS + CMB + FG + NOISE', fmt='.', color='blue')
ax_main.errorbar(ell_arr*0.995, cfn_mean[:lmax_ell_arr], yerr=cfn_std[:lmax_ell_arr], label='CMB + FG + NOISE', fmt='.', color='purple')
ax_main.plot(ell_arr, dl_in[:lmax_ell_arr], label='Fiducial CMB', color='black')
ax_main.errorbar(ell_arr*1.005, rmv_mean[:lmax_ell_arr], yerr=rmv_std[:lmax_ell_arr], label='Template Fitting method', fmt='.', color='green')
ax_main.errorbar(ell_arr*1.015, ps_mask_mean[:lmax_ell_arr], yerr=ps_mask_std[:lmax_ell_arr], label='Mask on QU', fmt='.', color='orange')
ax_main.errorbar(ell_arr*1.025, inp_mean[:lmax_ell_arr], yerr=inp_std[:lmax_ell_arr], label='Recycling + Inpaint on B', fmt='.', color='red')
# ax_main.plot(l, l*(l+1)*cl_cmb[2,:lmax_eff+1]/(2*np.pi), label='CMB input', color='black')


# Set labels and title for the main plot
ax_main.set_ylabel('$D_\\ell^{BB} [\mu K^2]$')
ax_main.set_xlim(58, lmax_eff)
ax_main.set_ylim(1e-3, 2e-1)
ax_main.set_title(f'Debiased power spectra {freq}GHz')
ax_main.legend()

res_pcfn = np.abs(pcfn_mean - dl_in)
res_cfn = np.abs(cfn_mean - dl_in)
res_rmv = np.abs(rmv_mean - dl_in)
res_inp = np.abs(inp_mean - dl_in)
res_ps_mask = np.abs(ps_mask_mean - dl_in)

res_line = mlines.Line2D([], [], color='black', linestyle='-', label='Residual')
std_line = mlines.Line2D([], [], color='black', linestyle=':', label='Std Deviation')

ax_sub.plot(ell_arr, res_pcfn[:lmax_ell_arr], label='PS + CMB + FG + NOISE', marker='.', color='blue')
ax_sub.plot(ell_arr, res_cfn[:lmax_ell_arr], label='CMB + FG + NOISE', marker='.', color='purple')
ax_sub.plot(ell_arr, res_rmv[:lmax_ell_arr], label='Template Fitting method', marker='.', color='green')
ax_sub.plot(ell_arr, res_inp[:lmax_ell_arr], label='Recycling + Inpaint on B', marker='.', color='red')
ax_sub.plot(ell_arr, res_ps_mask[:lmax_ell_arr], label='Mask on QU', marker='.', color='orange')

# Plot standard deviation in the subfigure (using scatter with no error bars)
ax_sub.plot(ell_arr, pcfn_std[:lmax_ell_arr], label='PS + CMB + FG + NOISE', color='blue', marker='.', linestyle=':')
ax_sub.plot(ell_arr, cfn_std[:lmax_ell_arr], label='CMB + FG + NOISE', color='purple', marker='.', linestyle=':')
ax_sub.plot(ell_arr, rmv_std[:lmax_ell_arr], label='Template Fitting method', color='green', marker='.', linestyle=':')
ax_sub.plot(ell_arr, ps_mask_std[:lmax_ell_arr], label='Mask on QU', color='orange', marker='.', linestyle=':')
ax_sub.plot(ell_arr, inp_std[:lmax_ell_arr], label='Recycling + Inpaint on B', color='red', marker='.', linestyle=':')

label_size = 10
ax_main.tick_params(axis='both', which='major', labelsize=label_size)
ax_main.tick_params(bottom=True, top=True, left=True, right=True, which = "major", direction="in", length=10, width=2);
ax_main.tick_params(bottom=True, top=True, left=True, right=True, which = "minor", direction="in", length=5, width=1.5);
# ax_main.grid(which='major', linestyle='-', linewidth=2)
# ax_main.grid(which='minor', linestyle='dashed', linewidth=0.9)
for axis in ['top','bottom','left','right']:
    ax_main.spines[axis].set_linewidth(2)

ax_sub.tick_params(axis='both', which='major', labelsize=label_size)
ax_sub.tick_params(bottom=True, top=True, left=True, right=True, which = "major", direction="in", length=10, width=2);
ax_sub.tick_params(bottom=True, top=True, left=True, right=True, which = "minor", direction="in", length=5, width=1.5);
# ax_sub.grid(which='major', linestyle='-', linewidth=2)
# ax_sub.grid(which='minor', linestyle='dashed', linewidth=0.9)
for axis in ['top','bottom','left','right']:
    ax_sub.spines[axis].set_linewidth(2)


# Set labels for the subfigure (only xlabel here)
ax_sub.set_xlabel('$\\ell$')
ax_sub.set_ylabel('Residual and Std Deviation')
# ax_sub.set_ylim(5e-4, 3e-2)
    # Set labels for the subfigure (only xlabel here)
ax_sub.set_xlabel('$\\ell$')
ax_sub.set_ylabel('Residual and Std Deviation')
ax_sub.legend(handles=[res_line, std_line])

# Adjust layout for better spacing
plt.tight_layout()
plt.subplots_adjust(hspace=0)

path_fig = Path('/afs/ihep.ac.cn/users/w/wangyiming25/tmp/20250219')
path_fig.mkdir(exist_ok=True, parents=True)
plt.savefig(path_fig / Path(f'{freq}GHz.png'), dpi=300)

# Show plot
plt.show()


