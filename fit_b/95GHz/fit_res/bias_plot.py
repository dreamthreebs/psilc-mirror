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
p_list = []

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

sim_mode = "STD"

rlz_range = np.arange(1, 200)
rlz_range_1k = np.arange(1, 5000)
base_path = f'./pcfn_dl4/{sim_mode}'

# cf_list = [np.load(f'{base_path}/cf/{rlz_idx}.npy') for rlz_idx in rlz_range]
n_qu_list = [np.load(f'{base_path}/n/{rlz_idx}.npy') for rlz_idx in rlz_range]

pcfn_list = [np.load(f'{base_path}/pcfn/{rlz_idx}.npy') for rlz_idx, n_qu in zip(rlz_range, n_qu_list)]
# cfn_list = [np.load(f'{base_path}/cfn/{rlz_idx}.npy') - n_qu for rlz_idx, n_qu in zip(rlz_range, n_qu_list)]
# cf_list = [cf for cf in cf_list]

rmv_bias_all_list = [
    np.load(f'./BIAS/rmv/bias_all_{rlz_idx}.npy')
    for rlz_idx in rlz_range
]

rmv_bias_model_list = [
    np.load(f'./BIAS/rmv/bias_model_{rlz_idx}.npy')
    for rlz_idx in rlz_range
]

inp_bias_all_list = [
    np.load(f'./BIAS/inp/bias_all_{rlz_idx}.npy')
    for rlz_idx in rlz_range
]

# print(f"{rmv_bias_all_list=}")
# print(f"{rmv_bias_model_list=}")

# ps_mask_list = [
#     np.load(f'./pcfn_dl4/PS_MASK_1/{sim_mode}/pcfn/{rlz_idx}.npy') -
#     np.load(f'./pcfn_dl4/PS_MASK_1/{sim_mode}/n/{rlz_idx}.npy')
#     for rlz_idx in rlz_range
# ]

# ps_mask_cf_list = [
#     np.load(f'./pcfn_dl4/PS_MASK_1/{sim_mode}/cf/{rlz_idx}.npy')
#     for rlz_idx in rlz_range
# ]

# inp_list = [
#     np.load(f'./pcfn_dl4/INP/{sim_mode}/{rlz_idx}.npy') -
#     np.load(f'./pcfn_dl4/INP/noise/{rlz_idx}.npy')
#     for rlz_idx in rlz_range
# ]


# print(f"{len(ps_mask_list)=}")
pcfn_mean = np.mean(pcfn_list, axis=0)
# cfn_mean = np.mean(cfn_list, axis=0)
# cf_mean = np.mean(cf_list, axis=0)

# rmv_mean = np.mean(rmv_list, axis=0)
rmv_bias_all_mean = np.mean(rmv_bias_all_list, axis=0)
rmv_bias_model_mean = np.mean(rmv_bias_model_list, axis=0)
inp_bias_all_mean = np.mean(inp_bias_all_list, axis=0)
# ps_mask_mean = np.mean(ps_mask_list, axis=0)
# ps_mask_mean_1 = np.mean(ps_mask_list_1, axis=0)
# ps_mask_cf_mean = np.mean(ps_mask_cf_list, axis=0)
# inp_mean = np.mean(inp_list, axis=0)

# test_mean = np.mean(test_list, axis=0)

# pcfn_std = np.std(pcfn_list, axis=0)
# cfn_std = np.std(cfn_list, axis=0)
# cf_std = np.std(cf_list, axis=0)

# rmv_std = np.std(rmv_list, axis=0)
# ps_mask_std = np.std(ps_mask_list, axis=0)
# ps_mask_std_1 = np.std(ps_mask_list_1, axis=0)
# inp_std = np.std(inp_list, axis=0)
# # test_std = np.std(test_list, axis=0)

dl_ps = np.load(f'./BIAS/pcfn/0.npy')
dl_unresolved_ps = np.load(f'./BIAS/unresolved_ps/0.npy')
dl_mask = np.load(f'./BIAS/mask/bias_all_0.npy')
# dl_resolved_ps = np.load(f'./BIAS//0.npy')


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

print(f'{ell_arr.shape=}')
# print(f'{pcfn_std.shape=}')

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

plt.figure(1)
# plt.plot(ell_arr, pcfn_mean[:lmax_ell_arr], label='pcfn', marker='.')
plt.plot(ell_arr, dl_in[:lmax_ell_arr], label='CMB', marker='.', color='black')
plt.plot(ell_arr, dl_unresolved_ps[:lmax_ell_arr], label='unresolved ps', marker='.')
plt.plot(ell_arr, dl_ps[:lmax_ell_arr], label='all ps contribution', marker='.')
plt.plot(ell_arr, rmv_bias_model_mean[:lmax_ell_arr], label='bias rmv from model', marker='.')
plt.plot(ell_arr, rmv_bias_all_mean[:lmax_ell_arr], label='bias rmv model + unresolved ps', marker='.')
plt.plot(ell_arr, rmv_bias_all_mean[:lmax_ell_arr] - rmv_bias_model_mean[:lmax_ell_arr], label='bias rmv unresolved ps', marker='.')
plt.plot(ell_arr, inp_bias_all_mean[:lmax_ell_arr], label='bias inp', marker='.')
plt.plot(ell_arr, dl_mask[:lmax_ell_arr], label='bias mask', marker='.')
plt.loglog()
plt.xlabel(r"$\ell$")
plt.ylabel(r"$D_\ell^{BB}$")
plt.title(f"{freq=}GHz")
plt.legend()
plt.show()


## Create figure with 2 subplots (main and subfigure), sharing the x-axis
#fig, ax_main= plt.subplots(figsize=(10, 8))

#s = 5

## Set the y-axis to logarithmic scale for both the main plot and subfigure
## ax_main.set_yscale('log')
#ax_main.set_xscale('log')


## Plot mean values in the main axis (no error bars here)
## ax_main.errorbar(ell_arr*0.985, pcfn_mean[:lmax_ell_arr], yerr=pcfn_std[:lmax_ell_arr], label='with-PS baseline', fmt='.', color='blue')
## ax_main.errorbar(ell_arr*0.995, cfn_mean[:lmax_ell_arr], yerr=cfn_std[:lmax_ell_arr], label='no-PS baseline', fmt='.', color='purple')
## ax_main.plot(ell_arr, dl_in[:lmax_ell_arr], label='Fiducial CMB', color='black')
## ax_main.errorbar(ell_arr*1.005, rmv_mean[:lmax_ell_arr], yerr=rmv_std[:lmax_ell_arr], label='TF', fmt='.', color='green')
## ax_main.errorbar(ell_arr*1.015, ps_mask_mean[:lmax_ell_arr], yerr=ps_mask_std[:lmax_ell_arr], label='M-QU', fmt='.', color='orange')
## ax_main.errorbar(ell_arr*1.025, inp_mean[:lmax_ell_arr], yerr=inp_std[:lmax_ell_arr], label='RI-B', fmt='.', color='red')
## ax_main.plot(l, l*(l+1)*cl_cmb[2,:lmax_eff+1]/(2*np.pi), label='CMB input', color='black')


## Set labels and title for the main plot
#ax_main.set_ylabel('$(D_\\ell^{BB}-D_\\ell^{BB FG,CMB,NOISE} )/ D_\\ell^{input}$')
#ax_main.set_xlim(58, lmax_eff)
#ax_main.set_ylim(-0.6,0.6)
#ax_main.set_title(f'relative bias and std')

#res_pcfn = np.abs(pcfn_mean - dl_in)
#res_cfn = np.abs(cfn_mean - dl_in)
#res_rmv = np.abs(rmv_mean - dl_in)
#res_inp = np.abs(inp_mean - dl_in)
#res_ps_mask = np.abs(ps_mask_mean - dl_in)

## res_line = mlines.Line2D([], [], color='black', linestyle='-', label='Residual')
## std_line = mlines.Line2D([], [], color='black', linestyle=':', label='Std Deviation')

#ax_main.plot(ell_arr, (pcfn_mean[:lmax_ell_arr]-cf_mean[:lmax_ell_arr])/dl_in[:lmax_ell_arr], label='with-PS baseline ps bias', color='blue')
#ax_main.plot(ell_arr, (cfn_mean[:lmax_ell_arr]-cf_mean[:lmax_ell_arr])/dl_in[:lmax_ell_arr], label='no-PS baseline ps bias', color='purple')
#ax_main.plot(ell_arr, (rmv_mean[:lmax_ell_arr]-cf_mean[:lmax_ell_arr])/dl_in[:lmax_ell_arr], label='TF ps bias', color='green')
#ax_main.plot(ell_arr, (ps_mask_mean[:lmax_ell_arr] - ps_mask_cf_mean[:lmax_ell_arr])/dl_in[:lmax_ell_arr], label='Masking ps bias', color='orange')
#ax_main.plot(ell_arr, (ps_mask_mean_1[:lmax_ell_arr]-cf_mean[:lmax_ell_arr])/dl_in[:lmax_ell_arr], label='Masking 2 degree ps bias', color='black')
#ax_main.plot(ell_arr, (inp_mean[:lmax_ell_arr]-cf_mean[:lmax_ell_arr])/dl_in[:lmax_ell_arr], label='Inpainting ps bias', color='red')
#ax_main.plot(ell_arr, dl_ps[:lmax_ell_arr]/dl_in[:lmax_ell_arr], label='ps bias', color='pink')
#ax_main.plot(ell_arr, dl_unresolved_ps[:lmax_ell_arr]/dl_in[:lmax_ell_arr], label='unresolved ps bias', color='yellow')
## ax_main.plot(ell_arr, (test_mean[:lmax_ell_arr])/dl_in[:lmax_ell_arr], label='fsky smaller ps bias', color='grey')

## ax_main.plot(ell_arr, pcfn_std[:lmax_ell_arr]/dl_in[:lmax_ell_arr], label='with-PS baseline std', color='blue', linestyle='--')
#ax_main.plot(ell_arr, cfn_std[:lmax_ell_arr]/dl_in[:lmax_ell_arr], color='purple', linestyle='--')
#ax_main.plot(ell_arr, -cfn_std[:lmax_ell_arr]/dl_in[:lmax_ell_arr], color='purple', linestyle='--')
#ax_main.fill_between(ell_arr, -cfn_std[:lmax_ell_arr]/dl_in[:lmax_ell_arr], cfn_std[:lmax_ell_arr]/dl_in[:lmax_ell_arr], label='no-PS baseline std', alpha=0.1, color='purple')
#ax_main.plot(ell_arr, ps_mask_std[:lmax_ell_arr]/dl_in[:lmax_ell_arr], label='M-QU std ps mask 1 degree', color='orange', linestyle='--')
#ax_main.plot(ell_arr, ps_mask_std_1[:lmax_ell_arr]/dl_in[:lmax_ell_arr], label='M-QU std ps mask 2 degree', color='grey', linestyle='--')
## ax_main.plot(ell_arr, rmv_std[:lmax_ell_arr]/dl_in[:lmax_ell_arr], label='TF std', color='green', linestyle='--')
## ax_main.plot(ell_arr, inp_std[:lmax_ell_arr]/dl_in[:lmax_ell_arr], label='RI-B std', color='red', linestyle='--')
#ax_main.legend()


## ax_main.plot(ell_arr, np.abs(pcfn_mean[:lmax_ell_arr]-cf_mean[:lmax_ell_arr]), label='with-PS baseline ps bias', color='blue')
## ax_main.plot(ell_arr, np.abs(cfn_mean[:lmax_ell_arr]-cf_mean[:lmax_ell_arr]), label='no-PS baseline ps bias', color='purple')
## ax_main.plot(ell_arr, np.abs(rmv_mean[:lmax_ell_arr]-cf_mean[:lmax_ell_arr]), label='TF ps bias', color='green')
## ax_main.plot(ell_arr, np.abs(ps_mask_mean[:lmax_ell_arr]-cf_mean[:lmax_ell_arr]), label='M-QU ps bias', color='orange')
## ax_main.plot(ell_arr, np.abs(inp_mean[:lmax_ell_arr]-cf_mean[:lmax_ell_arr]), label='RI-B ps bias', color='red')
## ax_main.plot(ell_arr, pcfn_std[:lmax_ell_arr], label='with-PS baseline std', color='blue', linestyle='--')
## ax_main.plot(ell_arr, cfn_std[:lmax_ell_arr], label='no-PS baseline std', color='purple', linestyle='--')
## ax_main.plot(ell_arr, ps_mask_std[:lmax_ell_arr], label='M-QU std', color='orange', linestyle='--')
## ax_main.plot(ell_arr, rmv_std[:lmax_ell_arr], label='TF std', color='green', linestyle='--')
## ax_main.plot(ell_arr, inp_std[:lmax_ell_arr], label='RI-B std', color='red', linestyle='--')

#ax_main.legend()
##
##
#label_size = 10
#ax_main.tick_params(axis='both', which='major', labelsize=label_size)
#ax_main.tick_params(bottom=True, top=True, left=True, right=True, which = "major", direction="in", length=10, width=2);
#ax_main.tick_params(bottom=True, top=True, left=True, right=True, which = "minor", direction="in", length=5, width=1.5);
## ax_main.grid(which='major', linestyle='-', linewidth=2)
## ax_main.grid(which='minor', linestyle='dashed', linewidth=0.9)
#for axis in ['top','bottom','left','right']:
#    ax_main.spines[axis].set_linewidth(2)

## Adjust layout for better spacing
#plt.tight_layout()
#plt.subplots_adjust(hspace=0)

## path_fig = Path('/afs/ihep.ac.cn/users/w/wangyiming25/tmp/20250323')
## path_fig.mkdir(exist_ok=True, parents=True)
## plt.savefig(path_fig / Path(f'{freq}GHz.png'), dpi=300)

## Show plot
#plt.show()


