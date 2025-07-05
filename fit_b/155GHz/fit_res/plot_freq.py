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
masking_list = []

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

        # n_ps_mask = np.load(f'./pcfn_dl4/MASK/noise/{rlz_idx}.npy')
        # ps_mask = np.load(f'./pcfn_dl4/MASK/STD/{rlz_idx}.npy') - n_ps_mask

        n_inp = np.load(f'./pcfn_dl4/INP/noise/{rlz_idx}.npy')
        inp = np.load(f'./pcfn_dl4/INP/{sim_mode}/{rlz_idx}.npy') - n_inp

        n_masking = np.load(f"./pcfn_dl4/MASK/noise/{rlz_idx}.npy")
        masking = np.load(f"./pcfn_dl4/MASK/STD/{rlz_idx}.npy")

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
        masking_list.append(masking)


    pcfn_mean = np.mean(pcfn_list, axis=0)
    cfn_mean = np.mean(cfn_list, axis=0)
    cf_mean = np.mean(cf_list, axis=0)

    rmv_mean = np.mean(rmv_list, axis=0)
    ps_mask_mean = np.mean(ps_mask_list, axis=0)
    inp_mean = np.mean(inp_list, axis=0)
    masking_mean = np.mean(masking_list, axis=0)

    pcfn_std = np.std(pcfn_list, axis=0)
    cfn_std = np.std(cfn_list, axis=0)
    cf_std = np.std(cf_list, axis=0)

    rmv_std = np.std(rmv_list, axis=0)
    ps_mask_std = np.std(ps_mask_list, axis=0)
    inp_std = np.std(inp_list, axis=0)
    masking_std = np.std(masking_list, axis=0)

    return pcfn_mean, cfn_mean, cf_mean, rmv_mean, ps_mask_mean, inp_mean, masking_mean, pcfn_std, cfn_std, cf_std, rmv_std, ps_mask_std, inp_std, masking_std

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
pcfn_mean, cfn_mean, cf_mean, rmv_mean, ps_mask_mean, inp_mean, masking_mean, pcfn_std, cfn_std, cf_std, rmv_std, ps_mask_std, inp_std, masking_std = mean_and_std(sim_mode='STD')

print(f'{ell_arr.shape=}')
print(f'{pcfn_std.shape=}')

# plt.figure(2)
# plt.scatter(ell_arr, pcfn_std, label='pcfn', marker='.')
# plt.scatter(ell_arr, cfn_std, label='cfn', marker='.')
# plt.scatter(ell_arr, cf_std, label='cf', marker='.')
# plt.scatter(ell_arr, rmv_std, label='rmv', marker='.')
# plt.scatter(ell_arr, ps_mask_std, label='ps_mask', marker='.')
# plt.scatter(ell_arr, inp_std, label='inp', marker='.')
# plt.xlabel('$\\ell$')
# plt.ylabel('$D_\\ell^{BB} [\mu K^2]$')

# plt.loglog()
# plt.legend()
# plt.title('standard deviation')
# plt.show()

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


res_pcfn = pcfn_mean - cfn_mean
res_cfn = cfn_mean - cfn_mean
res_rmv = rmv_mean - cfn_mean
res_inp = inp_mean - cfn_mean
res_ps_mask = ps_mask_mean - cfn_mean
res_masking = masking_mean - cfn_mean

# === Figure 1: Debiased Power Spectra with Errorbars ===
fig1, ax1 = plt.subplots(figsize=(6, 5))
s = 0

ax1.set_xscale('log')
ax1.set_yscale('log')

# Plot mean ± std
ax1.errorbar(ell_arr * 0.985, pcfn_mean[:lmax_ell_arr], yerr=pcfn_std[:lmax_ell_arr], label='Simulation with PS', fmt='.', color='blue', capsize=s)
ax1.errorbar(ell_arr * 1.00, cfn_mean[:lmax_ell_arr], yerr=cfn_std[:lmax_ell_arr], label='Simulation without PS', fmt='o', color='black', capsize=s, linestyle='-', markersize=3)
ax1.errorbar(ell_arr * 1.015, rmv_mean[:lmax_ell_arr], yerr=rmv_std[:lmax_ell_arr], label='GLSPF', fmt='.', color='green', capsize=s)
# ax1.errorbar(ell_arr * 1.03, ps_mask_mean[:lmax_ell_arr], yerr=ps_mask_std[:lmax_ell_arr], label='Masking', fmt='.', color='orange', capsize=s)
ax1.errorbar(ell_arr * 1.03, inp_mean[:lmax_ell_arr], yerr=inp_std[:lmax_ell_arr], label='Inpainting', fmt='.', color='red', capsize=s)

# Labels and layout
ax1.set_xlabel('$\\ell$', fontsize=14)
ax1.set_ylabel('$D_\\ell^{BB} [\mu K^2]$', fontsize=14)
ax1.set_xlim(55, lmax_eff * 1.01)
# ax1.set_title(f'Noise-debiased power spectra at {freq}GHz')
ax1.legend(loc='upper left', fontsize=11)

ax1.tick_params(axis='both', labelsize=12, direction="in")
ax1.tick_params(bottom=True, top=True, left=True, right=True, which = "major", direction="in", length=10, width=2);
ax1.tick_params(bottom=True, top=True, left=True, right=True, which = "minor", direction="in", length=5, width=1.5);
ax1.grid(which='major', linestyle='-', linewidth=2)
ax1.grid(which='minor', linestyle='dashed', linewidth=0.9)
for axis in ['top', 'bottom', 'left', 'right']:
    ax1.spines[axis].set_linewidth(2)

plt.tight_layout()
plt.subplots_adjust(hspace=0)
path_fig = Path('/afs/ihep.ac.cn/users/w/wangyiming25/tmp/20250609')
path_fig.mkdir(exist_ok=True, parents=True)
plt.savefig(path_fig / Path(f'power_{freq}GHz.png'), dpi=300)
plt.show()

# === Figure 2: Residual and Std in 2 Subplots (2x1) ===
fig2, axs = plt.subplots(2, 1, figsize=(6, 5), sharex=True, gridspec_kw={"hspace": 0})

# Compute residuals and ratios
residuals = {
    'PS + CMB + FG + NOISE': res_pcfn,
    'Template Fitting method': res_rmv,
    'CMB + FG + NOISE': res_cfn,
    'Recycling + Inpaint on B': res_inp,
    # 'Mask on QU': res_ps_mask,
}

stds = {
    'PS + CMB + FG + NOISE': pcfn_std,
    'Template Fitting method': rmv_std,
    'CMB + FG + NOISE': cfn_std,
    'Recycling + Inpaint on B': inp_std,
    # 'Mask on QU': ps_mask_std,
}

colors = {
    'PS + CMB + FG + NOISE': 'blue',
    'Template Fitting method': 'green',
    'CMB + FG + NOISE': 'black',
    'Recycling + Inpaint on B': 'red',
    # 'Mask on QU': 'orange',
}

# Top: Residuals
for label, data in residuals.items():
    axs[0].plot(ell_arr, data[:lmax_ell_arr] / cfn_mean[:lmax_ell_arr], label=label, marker='.', color=colors[label])

axs[0].set_ylabel('Residual $D_\\ell^{BB} / D_\\ell^{BB}$', fontsize=14)
# axs[0].legend(fontsize=8)
axs[0].tick_params(axis='both', labelsize=12, direction="in", width=2, length=4)
axs[0].tick_params(bottom=True, top=True, left=True, right=True, which = "major", direction="in", length=10, width=2);
axs[0].tick_params(bottom=True, top=True, left=True, right=True, which = "minor", direction="in", length=5, width=1.5);
axs[0].grid(which='major', linestyle='-', linewidth=2)
axs[0].set_ylim(-0.1,0.1)
axs[0].set_xscale('log')
for axis in ['top', 'bottom', 'left', 'right']:
    axs[0].spines[axis].set_linewidth(2)

# Bottom: Std deviations
for label, std in stds.items():
    axs[1].plot(ell_arr, std[:lmax_ell_arr] / cfn_mean[:lmax_ell_arr], label=label, marker='.', linestyle=':', color=colors[label])

axs[1].set_xlabel('$\\ell$', fontsize=14)
axs[1].set_ylabel(r'$\sigma(D_\ell^{BB}) / D_\ell^{BB}$', fontsize=14)
axs[1].tick_params(axis='both', labelsize=12, direction="in", width=2, length=4)
axs[1].tick_params(bottom=True, top=True, left=True, right=True, which = "major", direction="in", length=10, width=2);
axs[1].tick_params(bottom=True, top=True, left=True, right=True, which = "minor", direction="in", length=5, width=1.5);
axs[1].grid(which='major', linestyle='-', linewidth=2)
for axis in ['top', 'bottom', 'left', 'right']:
    axs[1].spines[axis].set_linewidth(2)

# axs[1].legend(fontsize=8)
axs[1].set_xscale('log')
axs[1].set_yscale('log')
axs[1].set_ylim(0,0.9)
fig2.align_ylabels(axs)

# Final layout
plt.tight_layout()

path_fig = Path('/afs/ihep.ac.cn/users/w/wangyiming25/tmp/20250609')
path_fig.mkdir(exist_ok=True, parents=True)
plt.savefig(path_fig / Path(f'res_std_{freq}GHz.png'), dpi=300)

plt.show()
