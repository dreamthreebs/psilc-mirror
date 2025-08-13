import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pymaster as nmt

from pathlib import Path

beam = 17
lmax = 1500

rlz_idx = 0

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

def calc_dl_from_scalar_map(scalar_map, bl, apo_mask, bin_dl, masked_on_input):
    scalar_field = nmt.NmtField(apo_mask, [scalar_map], beam=bl, masked_on_input=masked_on_input, lmax=lmax, lmax_mask=lmax)
    dl = nmt.compute_full_master(scalar_field, scalar_field, bin_dl)
    return dl[0]

def calc_dl4():
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax, pol=True)[:,2]
    l_min_edges, l_max_edges = generate_bins(l_min_start=42, delta_l_min=40, l_max=lmax+1, fold=0.1, l_threshold=400)
    # delta_ell = 30
    # bin_dl = nmt.NmtBin.from_nside_linear(nside, nlb=delta_ell, is_Dell=True)
    # bin_dl = nmt.NmtBin.from_lmax_linear(lmax=lmax, nlb=40, is_Dell=True)
    bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    ell_arr = bin_dl.get_effective_ells()

    apo_mask = np.load(f'../../psfit/fitv4/fit_res/2048/ps_mask/new_mask/apo_C1_3_apo_3_apo_3.npy')


    # m = np.load(f'./data/mean/pcfn/{rlz_idx}.npy')

    def _calc_dl(sim_mode, method):
        _map = np.load(f'./data2/{sim_mode}/{method}/{rlz_idx}.npy')

        dl = calc_dl_from_scalar_map(scalar_map=_map, bl=bl, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=False)
        path_dl = Path(f'./dl_res4/{sim_mode}/{method}')
        path_dl.mkdir(exist_ok=True, parents=True)
        np.save(path_dl / Path(f'{rlz_idx}.npy'), dl)

        if method == 'cf':
            pass
        else:
            n_map = np.load(f'./data2/{sim_mode}/n_{method}/{rlz_idx}.npy')
            dl_n = calc_dl_from_scalar_map(scalar_map=n_map, bl=bl, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=False)
            path_dl_n = Path(f'./dl_res4/{sim_mode}/n_{method}')
            path_dl_n.mkdir(exist_ok=True, parents=True)
            np.save(path_dl_n / Path(f'{rlz_idx}.npy'), dl_n)
        print(f'dl {sim_mode} {method} is ok')

    _calc_dl(sim_mode='std', method='pcfn')
    _calc_dl(sim_mode='std', method='cfn')
    _calc_dl(sim_mode='std', method='rmv')
    # _calc_dl(sim_mode='std', method='cf')

    _calc_dl(sim_mode='std', method='inp')

def get_mean_std():
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax, pol=True)[:,2]
    l_min_edges, l_max_edges = generate_bins(l_min_start=30, delta_l_min=30, l_max=lmax+1, fold=0.2)
    # delta_ell = 30
    # bin_dl = nmt.NmtBin.from_nside_linear(nside, nlb=delta_ell, is_Dell=True)
    # bin_dl = nmt.NmtBin.from_lmax_linear(lmax=lmax, nlb=30, is_Dell=True)
    bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    ell_arr = bin_dl.get_effective_ells()
    cl = np.load(f'../../src/cmbsim/cmbdata/cmbcl_8k.npy')[:lmax+1,2]
    print(f'{cl.shape=}')
    l = np.arange(len(cl))


    pcfn_mean = np.mean([np.load(f'./dl_res/std/pcfn/{rlz_idx}.npy') - np.load(f'./dl_res/std/n_pcfn/{rlz_idx}.npy') for rlz_idx in np.arange(1,200)], axis=0)
    pcfn_std = np.std([np.load(f'./dl_res/std/pcfn/{rlz_idx}.npy') - np.load(f'./dl_res/std/n_pcfn/{rlz_idx}.npy') for rlz_idx in np.arange(1,200)], axis=0)
    cfn_mean = np.mean([np.load(f'./dl_res/std/cfn/{rlz_idx}.npy') - np.load(f'./dl_res/std/n_cfn/{rlz_idx}.npy') for rlz_idx in np.arange(1,200)], axis=0)
    cfn_std = np.std([np.load(f'./dl_res/std/cfn/{rlz_idx}.npy') - np.load(f'./dl_res/std/n_cfn/{rlz_idx}.npy') for rlz_idx in np.arange(1,200)], axis=0)
    rmv_mean = np.mean([np.load(f'./dl_res/std/rmv/{rlz_idx}.npy') - np.load(f'./dl_res/std/n_rmv/{rlz_idx}.npy') for rlz_idx in np.arange(1,200)], axis=0)
    rmv_std = np.std([np.load(f'./dl_res/std/rmv/{rlz_idx}.npy') - np.load(f'./dl_res/std/n_rmv/{rlz_idx}.npy') for rlz_idx in np.arange(1,200)], axis=0)
    inp_mean = np.mean([np.load(f'./dl_res/std/inp/{rlz_idx}.npy') - np.load(f'./dl_res/std/n_inp/{rlz_idx}.npy') for rlz_idx in np.arange(1,200)], axis=0)
    inp_std = np.std([np.load(f'./dl_res/std/inp/{rlz_idx}.npy') - np.load(f'./dl_res/std/n_inp/{rlz_idx}.npy') for rlz_idx in np.arange(1,200)], axis=0)
    print(f'mean std over')

    fig, ax = plt.subplots()
    
    # Set y-axis to logarithmic scale
    ax.set_yscale('log')
    ax.set_xscale('log')
    
    # Error bars (only in y, in log scale)
    ax.errorbar(ell_arr, pcfn_mean, yerr=pcfn_std, fmt='.', capsize=5, label='PS + CMB + FG + NOISE')
    ax.errorbar(ell_arr+1.2, cfn_mean, yerr=cfn_std, fmt='.', capsize=5, label='CMB + FG + NOISE')
    ax.errorbar(ell_arr+2.4, rmv_mean, yerr=rmv_std, fmt='.', capsize=5, label='Template Fitting method')
    ax.errorbar(ell_arr+3.6, inp_mean, yerr=inp_std, fmt='.', capsize=5, label='Recycling + inpaint on B')
    ax.loglog(l, l*(l+1)*cl/(2*np.pi), label='cmb input')
    
    # Add labels, legend, and title
    ax.set_xlabel('$\\ell$')
    ax.set_ylabel('$D_\\ell^{BB} [\mu K^2]$')
    # ax.set_xlim(2,lmax_eff)
    # ax.set_ylim(1e-2,5e0)
    ax.set_title('Debiased power spectra and standard deviation')
    ax.legend()
    plt.show()

def plot_nilc_ref():
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax, pol=True)[:,2]
    l_min_edges, l_max_edges = generate_bins(l_min_start=42, delta_l_min=40, l_max=lmax+1, fold=0.1, l_threshold=400)
    # delta_ell = 30
    # bin_dl = nmt.NmtBin.from_nside_linear(nside, nlb=delta_ell, is_Dell=True)
    # bin_dl = nmt.NmtBin.from_lmax_linear(lmax=lmax, nlb=40, is_Dell=True)
    bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    ell_arr = bin_dl.get_effective_ells()
    print(f'{ell_arr=}')

    cl = np.load(f'../../src/cmbsim/cmbdata/cmbcl_8k.npy')[:lmax+1,2]
    print(f'{cl.shape=}')
    l = np.arange(len(cl))

    pcfn_mean = np.mean([np.load(f'./dl_res4/std/pcfn/{rlz_idx}.npy') - np.load(f'./dl_res4/std/n_pcfn/{rlz_idx}.npy') for rlz_idx in np.arange(1,200)], axis=0)
    n_mean = np.mean([np.load(f'./dl_res4/std/n_pcfn/{rlz_idx}.npy') for rlz_idx in np.arange(1,200)], axis=0)
    pcfn_std = np.std([np.load(f'./dl_res4/std/pcfn/{rlz_idx}.npy') - np.load(f'./dl_res4/std/n_pcfn/{rlz_idx}.npy') for rlz_idx in np.arange(1,200)], axis=0)
    cfn_mean = np.mean([np.load(f'./dl_res4/std/cfn/{rlz_idx}.npy') - np.load(f'./dl_res4/std/n_cfn/{rlz_idx}.npy') for rlz_idx in np.arange(1,200)], axis=0)
    cfn_std = np.std([np.load(f'./dl_res4/std/cfn/{rlz_idx}.npy') - np.load(f'./dl_res4/std/n_cfn/{rlz_idx}.npy') for rlz_idx in np.arange(1,200)], axis=0)
    rmv_mean = np.mean([np.load(f'./dl_res4/std/rmv/{rlz_idx}.npy') - np.load(f'./dl_res4/std/n_rmv/{rlz_idx}.npy') for rlz_idx in np.arange(1,200)], axis=0)
    rmv_std = np.std([np.load(f'./dl_res4/std/rmv/{rlz_idx}.npy') - np.load(f'./dl_res4/std/n_rmv/{rlz_idx}.npy') for rlz_idx in np.arange(1,200)], axis=0)
    inp_mean = np.mean([np.load(f'./dl_res4/std/inp/{rlz_idx}.npy') - np.load(f'./dl_res4/std/n_inp/{rlz_idx}.npy') for rlz_idx in np.arange(1,200)], axis=0)
    inp_std = np.std([np.load(f'./dl_res4/std/inp/{rlz_idx}.npy') - np.load(f'./dl_res4/std/n_inp/{rlz_idx}.npy') for rlz_idx in np.arange(1,200)], axis=0)
    masking_30_mean = np.mean([np.load(f'./dl_res4/mask/pcfn_270_{rlz_idx}.npy') - np.load(f'./dl_res4/mask/n_270_{rlz_idx}.npy') for rlz_idx in np.arange(1,200)], axis=0)
    masking_30_std = np.std([np.load(f'./dl_res4/mask/pcfn_270_{rlz_idx}.npy') - np.load(f'./dl_res4/mask/n_270_{rlz_idx}.npy') for rlz_idx in np.arange(1,200)], axis=0)
    masking_95_mean = np.mean([np.load(f'./dl_res4/mask/pcfn_union_{rlz_idx}.npy') - np.load(f'./dl_res4/mask/n_union_{rlz_idx}.npy') for rlz_idx in np.arange(1,200)], axis=0)
    masking_95_std = np.std([np.load(f'./dl_res4/mask/pcfn_union_{rlz_idx}.npy') - np.load(f'./dl_res4/mask/n_union_{rlz_idx}.npy') for rlz_idx in np.arange(1,200)], axis=0)


    print(f'mean std over')
    fid_mean = np.mean([np.load(f'./dl_res4/fid_cmb/{rlz_idx}.npy') for rlz_idx in np.arange(1,200)], axis=0)
    fid_std = np.std([np.load(f'./dl_res4/fid_cmb/{rlz_idx}.npy') for rlz_idx in np.arange(1,200)], axis=0)
    # cf_mean = np.mean([np.load(f'./dl_res4/std/cf/{rlz_idx}.npy') for rlz_idx in np.arange(1,200)], axis=0)

    # bias_std_2_pcfn = np.sqrt((pcfn_mean-fid_mean)**2 + pcfn_std**2)
    # bias_std_2_cfn = np.sqrt((cfn_mean-fid_mean)**2 + cfn_std**2)
    # bias_std_2_rmv = np.sqrt((rmv_mean-fid_mean)**2 + rmv_std**2)
    # bias_std_2_inp= np.sqrt((inp_mean-fid_mean)**2 + inp_std**2)

    # dl_in = bin_dl.bin_cell(cls_in=cl)
    dl_in = np.load(f'./dl_res4/mask/th_apo_0.npy')
    dl_in_ps = np.load(f'./dl_res4/mask/th_union_0.npy')
    # dl_in_ps_3deg = np.load(f'./dl_res4/mask/th_95_1deg_0.npy')
    dl_in_ps_bin = np.load(f'./dl_res4/mask/th_no_north_0.npy')

    res_pcfn = np.abs(pcfn_mean - dl_in)
    res_cfn = np.abs(cfn_mean - dl_in)
    res_rmv = np.abs(rmv_mean - dl_in)
    res_inp = np.abs(inp_mean - dl_in)
    res_masking_30 = np.abs(masking_30_mean - dl_in_ps_bin)
    res_masking_95 = np.abs(masking_95_mean - dl_in_ps)

    res_line = mlines.Line2D([], [], color='black', linestyle='-', label='Residual')
    std_line = mlines.Line2D([], [], color='black', linestyle=':', label='Std Deviation')

    # Create figure with 2 subplots (main and subfigure), sharing the x-axis
    fig, (ax_main, ax_sub) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    lmax_ell_arr = len(ell_arr) - 2
    ell_arr = ell_arr[:-2]

    # Set the y-axis to logarithmic scale for both the main plot and subfigure
    ax_main.set_yscale('log')
    ax_sub.set_yscale('log')
    ax_main.set_xscale('log')
    ax_sub.set_xscale('log')


    # Plot mean values in the main axis (no error bars here)
    s = 3
    mk_s = 5
    cap_s = 3

    # ax_main.errorbar(ell_arr, fid_mean[:lmax_ell_arr], yerr=fid_std[:lmax_ell_arr], capsize=s, label='Fiducial CMB', color='black', markersize=mk_s)
    ax_main.plot(ell_arr, dl_in[:lmax_ell_arr], label='Fiducial CMB apo', color='black', linewidth=2.5)
    # ax_main.plot(ell_arr, dl_in_ps[:lmax_ell_arr], label='Fiducial CMB ps 2deg', color='grey', linewidth=2.5)
    # ax_main.plot(ell_arr, dl_in_ps_3deg[:lmax_ell_arr], label='Fiducial CMB ps 1deg', linewidth=2.5)
    ax_main.plot(ell_arr, dl_in_ps_bin[:lmax_ell_arr], label='Fiducial CMB ps bin', linewidth=2.5)
    ax_main.plot(ell_arr, dl_in_ps[:lmax_ell_arr], label='Fiducial CMB ps union', linewidth=2.5)
    ax_main.errorbar(ell_arr*0.985, pcfn_mean[:lmax_ell_arr], yerr=pcfn_std[:lmax_ell_arr], fmt='.', capsize=s, label='with-PS baseline', markersize=mk_s, color='blue')
    ax_main.errorbar(ell_arr*0.995, cfn_mean[:lmax_ell_arr], yerr=cfn_std[:lmax_ell_arr], fmt='.', capsize=s, label='no-PS baseline', markersize=mk_s, color='purple')
    ax_main.errorbar(ell_arr*1.005, rmv_mean[:lmax_ell_arr], yerr=rmv_std[:lmax_ell_arr], fmt='.', capsize=s, label='TF', markersize=mk_s, color='green')
    ax_main.errorbar(ell_arr*1.015, inp_mean[:lmax_ell_arr], yerr=inp_std[:lmax_ell_arr], fmt='.', capsize=s, label='RI-B', markersize=mk_s, color='red')
    ax_main.errorbar(ell_arr*1.025, masking_30_mean[:lmax_ell_arr], yerr=masking_30_std[:lmax_ell_arr], fmt='.', capsize=s, label='M-QU 30 2deg', markersize=mk_s, color='orange')
    ax_main.errorbar(ell_arr*1.035, masking_95_mean[:lmax_ell_arr], yerr=masking_95_std[:lmax_ell_arr], fmt='.', capsize=s, label='M-QU 95 2deg', markersize=mk_s, color='yellow', linestyle='--')

    # Set labels and title for the main plot
    ax_main.set_ylabel('$D_\\ell^{BB} [\mu K^2]$')
    ax_main.set_xlim(58, 1290)
    # ax_main.set_ylim(1e-4, 2e-1)
    ax_main.set_title('Debiased power spectra')
    ax_main.legend()


    # Plot standard deviation in the subfigure (using scatter with no error bars)

    # ax_sub.scatter(ell_arr, fid_std[:lmax_ell_arr], s=s, label='Fiducial CMB', color='black', marker='.')
    # ax_sub.scatter(ell_arr+5.0, bias_std_2_pcfn[:lmax_ell_arr], s=s, label='PS + CMB + FG + NOISE', marker='.')
    # ax_sub.scatter(ell_arr+10.0, bias_std_2_cfn[:lmax_ell_arr], s=s, label='CMB + FG + NOISE', marker='.')
    # ax_sub.scatter(ell_arr+15.0, bias_std_2_rmv[:lmax_ell_arr], s=s, label='Template Fitting method', marker='.')
    # ax_sub.scatter(ell_arr+20.0, bias_std_2_inp[:lmax_ell_arr], s=s, label='Recycling + Inpaint on B', marker='.')

    # ax_sub.plot(ell_arr, fid_std[:lmax_ell_arr], s=s, label='Fiducial CMB', color='black', marker='.')

    ax_sub.plot(ell_arr, res_pcfn[:lmax_ell_arr], label='with-PS baseline', marker='.', color='blue')
    ax_sub.plot(ell_arr, res_cfn[:lmax_ell_arr], label='no-PS baseline', marker='.', color='purple')
    ax_sub.plot(ell_arr, res_rmv[:lmax_ell_arr], label='TF', marker='.', color='green')
    ax_sub.plot(ell_arr, res_inp[:lmax_ell_arr], label='RI-B', marker='.', color='red')
    ax_sub.plot(ell_arr, res_masking_30[:lmax_ell_arr], label='M-QU 30', marker='.', color='orange')
    ax_sub.plot(ell_arr, res_masking_95[:lmax_ell_arr], label='M-QU 95', marker='.', color='yellow')

    ax_sub.plot(ell_arr, pcfn_std[:lmax_ell_arr], label='with-PS baseline', marker='.', linestyle=':', color='blue')
    ax_sub.plot(ell_arr, cfn_std[:lmax_ell_arr], label='no-PS baseline', marker='.', linestyle=':', color='purple')
    ax_sub.plot(ell_arr, rmv_std[:lmax_ell_arr], label='TF', marker='.', linestyle=':', color='green')
    ax_sub.plot(ell_arr, inp_std[:lmax_ell_arr], label='RI-B', marker='.', linestyle=':', color='red')
    ax_sub.plot(ell_arr, masking_30_std[:lmax_ell_arr], label='M-QU 30', marker='.', linestyle=':', color='orange')
    ax_sub.plot(ell_arr, masking_95_std[:lmax_ell_arr], label='M-QU 95', marker='.', linestyle=':', color='yellow')

    
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
    ax_sub.set_ylabel('Residual and Std deviation')
    ax_sub.legend(handles=[res_line, std_line])

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)

    # plt.savefig(f'/afs/ihep.ac.cn/users/w/wangyiming25/tmp/20250323/nilc_res.png', dpi=300)

    # Show plot
    plt.show()


def plot_nilc():
    # TODO: add cmb mean and std!
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax, pol=True)[:,2]
    l_min_edges, l_max_edges = generate_bins(l_min_start=42, delta_l_min=40, l_max=lmax+1, fold=0.1, l_threshold=400)
    # delta_ell = 30
    # bin_dl = nmt.NmtBin.from_nside_linear(nside, nlb=delta_ell, is_Dell=True)
    # bin_dl = nmt.NmtBin.from_lmax_linear(lmax=lmax, nlb=40, is_Dell=True)
    bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    ell_arr = bin_dl.get_effective_ells()
    print(f'{ell_arr=}')

    cl = np.load(f'../../src/cmbsim/cmbdata/cmbcl_8k.npy')[:lmax+1,2]
    print(f'{cl.shape=}')
    l = np.arange(len(cl))

    pcfn_mean = np.mean([np.load(f'./dl_res4/std/pcfn/{rlz_idx}.npy') - np.load(f'./dl_res4/std/n_pcfn/{rlz_idx}.npy') for rlz_idx in np.arange(1,200)], axis=0)
    n_mean = np.mean([np.load(f'./dl_res4/std/n_pcfn/{rlz_idx}.npy') for rlz_idx in np.arange(1,200)], axis=0)
    pcfn_std = np.std([np.load(f'./dl_res4/std/pcfn/{rlz_idx}.npy') - np.load(f'./dl_res4/std/n_pcfn/{rlz_idx}.npy') for rlz_idx in np.arange(1,200)], axis=0)
    cfn_mean = np.mean([np.load(f'./dl_res4/std/cfn/{rlz_idx}.npy') - np.load(f'./dl_res4/std/n_cfn/{rlz_idx}.npy') for rlz_idx in np.arange(1,200)], axis=0)
    cfn_std = np.std([np.load(f'./dl_res4/std/cfn/{rlz_idx}.npy') - np.load(f'./dl_res4/std/n_cfn/{rlz_idx}.npy') for rlz_idx in np.arange(1,200)], axis=0)
    rmv_mean = np.mean([np.load(f'./dl_res4/std/rmv/{rlz_idx}.npy') - np.load(f'./dl_res4/std/n_rmv/{rlz_idx}.npy') for rlz_idx in np.arange(1,200)], axis=0)
    rmv_std = np.std([np.load(f'./dl_res4/std/rmv/{rlz_idx}.npy') - np.load(f'./dl_res4/std/n_rmv/{rlz_idx}.npy') for rlz_idx in np.arange(1,200)], axis=0)
    inp_mean = np.mean([np.load(f'./dl_res4/std/inp/{rlz_idx}.npy') - np.load(f'./dl_res4/std/n_inp/{rlz_idx}.npy') for rlz_idx in np.arange(1,200)], axis=0)
    inp_std = np.std([np.load(f'./dl_res4/std/inp/{rlz_idx}.npy') - np.load(f'./dl_res4/std/n_inp/{rlz_idx}.npy') for rlz_idx in np.arange(1,200)], axis=0)
    ps_mask_30_mean = np.mean([np.load(f'./dl_res4/mask/pcfn_union_new_apo_{rlz_idx}.npy') - np.load(f'./dl_res4/mask/n_union_new_apo_{rlz_idx}.npy') for rlz_idx in np.arange(1,200)], axis=0)
    ps_mask_30_std = np.std([np.load(f'./dl_res4/mask/pcfn_union_new_apo_{rlz_idx}.npy') - np.load(f'./dl_res4/mask/n_union_new_apo_{rlz_idx}.npy') for rlz_idx in np.arange(1,200)], axis=0)
    # ps_mask_mean = np.mean([np.load(f'./dl_res4/mask/pcfn_union_{rlz_idx}.npy') - np.load(f'./dl_res4/mask/n_union_{rlz_idx}.npy') for rlz_idx in np.arange(1,200)], axis=0)
    # ps_mask_std = np.std([np.load(f'./dl_res4/mask/pcfn_union_{rlz_idx}.npy') - np.load(f'./dl_res4/mask/n_union_{rlz_idx}.npy') for rlz_idx in np.arange(1,200)], axis=0)
    ps_mask_mean = np.mean([np.load(f'./dl_res4/mask/pcfn_30_67_{rlz_idx}.npy') - np.load(f'./dl_res4/mask/n_30_67_{rlz_idx}.npy') for rlz_idx in np.arange(1,200)], axis=0)
    ps_mask_std = np.std([np.load(f'./dl_res4/mask/pcfn_30_67_{rlz_idx}.npy') - np.load(f'./dl_res4/mask/n_30_67_{rlz_idx}.npy') for rlz_idx in np.arange(1,200)], axis=0)



    print(f'mean std over')
    fid_mean = np.mean([np.load(f'./dl_res4/fid_cmb/{rlz_idx}.npy') for rlz_idx in np.arange(1,200)], axis=0)
    fid_std = np.std([np.load(f'./dl_res4/fid_cmb/{rlz_idx}.npy') for rlz_idx in np.arange(1,200)], axis=0)
    # cf_mean = np.mean([np.load(f'./dl_res4/std/cf/{rlz_idx}.npy') for rlz_idx in np.arange(1,200)], axis=0)

    # bias_std_2_pcfn = np.sqrt((pcfn_mean-fid_mean)**2 + pcfn_std**2)
    # bias_std_2_cfn = np.sqrt((cfn_mean-fid_mean)**2 + cfn_std**2)
    # bias_std_2_rmv = np.sqrt((rmv_mean-fid_mean)**2 + rmv_std**2)
    # bias_std_2_inp= np.sqrt((inp_mean-fid_mean)**2 + inp_std**2)

    # dl_in = bin_dl.bin_cell(cls_in=cl)
    dl_in = np.load(f'./dl_res4/mask/th_apo_0.npy')
    # dl_in_ps = np.load(f'./dl_res4/mask/th_union_0.npy')
    dl_in_ps = np.load(f'./dl_res4/mask/th_30_67_0.npy')
    dl_in_ps_30GHz = np.load(f'./dl_res4/mask/th_union_new_apo_0.npy')

    dl_in_std = np.load(f'./dl_res4/mask/th_std_apo_new_0.npy')

    res_pcfn = pcfn_mean - dl_in
    res_cfn = cfn_mean - dl_in
    res_th = dl_in - dl_in
    res_rmv = rmv_mean - dl_in
    res_inp = inp_mean - dl_in
    # res_masking_30 = masking_30_mean - dl_in_ps_bin
    res_ps_mask = ps_mask_mean - dl_in_ps
    res_ps_mask_30 = ps_mask_30_mean - dl_in_ps_30GHz

    res_line = mlines.Line2D([], [], color='black', linestyle='-', label='Residual')
    std_line = mlines.Line2D([], [], color='black', linestyle=':', label='Std Deviation')

   
    # === Figure 1: Debiased Power Spectra with Errorbars ===
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    s = 0

    lmax_eff = calc_lmax(beam=beam)
    lmax_ell_arr = find_left_nearest_index_np(ell_arr, target=lmax_eff)
    lmax_ell_arr = len(ell_arr) - 2
    ell_arr = ell_arr[:lmax_ell_arr]

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # Plot mean ± std
    ax1.errorbar(ell_arr * 0.970, pcfn_mean[:lmax_ell_arr], yerr=pcfn_std[:lmax_ell_arr], label='Simulation with PS', fmt='.', color='blue', capsize=s)
    ax1.errorbar(ell_arr * 0.985, cfn_mean[:lmax_ell_arr], yerr=cfn_std[:lmax_ell_arr], label='Simulation without PS', fmt='.', color='purple', capsize=s)
    ax1.errorbar(ell_arr * 1.00, dl_in[:lmax_ell_arr], yerr=dl_in_std[:lmax_ell_arr], label='Fiducial CMB', fmt='o', color='black', capsize=s, linestyle='-', markersize=3)
    ax1.errorbar(ell_arr * 1.015, rmv_mean[:lmax_ell_arr], yerr=rmv_std[:lmax_ell_arr], label='GPSF', fmt='.', color='green', capsize=s)
    # ax1.errorbar(ell_arr * 1.00, dl_in_ps_30GHz[:lmax_ell_arr], yerr=dl_in_std[:lmax_ell_arr], label='Fiducial CMB union', fmt='o', color='grey', capsize=s, linestyle='-', markersize=3)
    ax1.errorbar(ell_arr * 1.03, inp_mean[:lmax_ell_arr], yerr=inp_std[:lmax_ell_arr], label='Inpainting', fmt='.', color='red', capsize=s)
    ax1.errorbar(ell_arr * 1.045, ps_mask_mean[:lmax_ell_arr], yerr=ps_mask_std[:lmax_ell_arr], label='Masking', fmt='.', color='orange', capsize=s)
    # ax1.errorbar(ell_arr * 1.060, ps_mask_30_mean[:lmax_ell_arr], yerr=ps_mask_30_std[:lmax_ell_arr], label='M-QU 30GHz mask', fmt='.', color='yellow', capsize=s)
    
    # Labels and layout
    ax1.set_xlabel('$\\ell$', fontsize=14)
    ax1.set_ylabel('$D_\\ell^{BB} [\mu K^2]$', fontsize=14)
    ax1.set_xlim(55, lmax_eff * 1.01)
    # ax1.set_title(f'Noise-debiased power spectra at {freq}GHz')
    ax1.legend(loc='upper left', fontsize=11)
    
    ax1.tick_params(axis='both', labelsize=12, direction="in")
    ax1.tick_params(bottom=True, top=True, left=True, right=True, which = "major", direction="in", length=10, width=2);
    ax1.tick_params(bottom=True, top=True, left=True, right=True, which = "minor", direction="in", length=5, width=1.5);
    ax1.grid(which='major', linestyle='-', linewidth=1, alpha=0.3)
    ax1.grid(which='minor', linestyle='dashed', linewidth=0.5, alpha=0.3)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax1.spines[axis].set_linewidth(2)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    path_fig = Path('/afs/ihep.ac.cn/users/w/wangyiming25/tmp/20250726')
    path_fig.mkdir(exist_ok=True, parents=True)
    plt.savefig(path_fig / Path(f'power_nilc.png'), dpi=300)
    plt.show()
    
    # === Figure 2: Residual and Std in 2 Subplots (2x1) ===
    fig2, axs = plt.subplots(2, 1, figsize=(6, 5), sharex=True, gridspec_kw={"hspace": 0})
    
    # Compute residuals and ratios
    residuals = {
        'PS + CMB + FG + NOISE': res_pcfn,
        'Template Fitting method': res_rmv,
        'CMB + FG + NOISE': res_cfn,
        'Fiducial CMB': res_th,
        'Recycling + Inpaint on B': res_inp,
        'Mask on B': res_ps_mask,
        # 'Mask on B union': res_ps_mask_30,
    }
    
    stds = {
        'PS + CMB + FG + NOISE': pcfn_std,
        'Template Fitting method': rmv_std,
        'CMB + FG + NOISE': cfn_std,
        'Fiducial CMB': dl_in_std,
        'Recycling + Inpaint on B': inp_std,
        'Mask on B': ps_mask_std,
        # 'Mask on B union': ps_mask_30_std,
    }
    
    colors = {
        'PS + CMB + FG + NOISE': 'blue',
        'Template Fitting method': 'green',
        'CMB + FG + NOISE': 'purple',
        'Fiducial CMB': 'black',
        'Recycling + Inpaint on B': 'red',
        'Mask on B': 'orange',
        # 'Mask on B union': 'yellow',
    }
    
    # Top: Residuals
    for label, data in residuals.items():
        if label == 'Mask on B':
            axs[0].plot(ell_arr, data[:lmax_ell_arr] / dl_in_ps[:lmax_ell_arr], label=label, marker='.', color=colors[label])
        elif label == 'Mask on B union':
            axs[0].plot(ell_arr, data[:lmax_ell_arr] / dl_in_ps_30GHz[:lmax_ell_arr], label=label, marker='.', color=colors[label])
        else:
            axs[0].plot(ell_arr, data[:lmax_ell_arr] / dl_in[:lmax_ell_arr], label=label, marker='.', color=colors[label])
    
    axs[0].set_ylabel('Residual $D_\\ell^{BB} / D_\\ell^{BB}$', fontsize=14)
    # axs[0].legend(fontsize=8)
    axs[0].tick_params(axis='both', labelsize=12, direction="in", width=2, length=4)
    axs[0].tick_params(bottom=True, top=True, left=True, right=True, which = "major", direction="in", length=10, width=2);
    axs[0].tick_params(bottom=True, top=True, left=True, right=True, which = "minor", direction="in", length=5, width=1.5);
    axs[0].grid(which='major', linestyle='-', linewidth=0.5, alpha=0.3)
    axs[0].set_ylim(-0.05,0.20)
    axs[0].set_xscale('log')
    for axis in ['top', 'bottom', 'left', 'right']:
        axs[0].spines[axis].set_linewidth(2)
    
    # Bottom: Std deviations
    for label, std in stds.items():
        if label == 'Mask on QU':
            axs[1].plot(ell_arr, std[:lmax_ell_arr] / dl_in_ps[:lmax_ell_arr], label=label, marker='.', linestyle=':', color=colors[label])
        else:
            axs[1].plot(ell_arr, std[:lmax_ell_arr] / dl_in[:lmax_ell_arr], label=label, marker='.', linestyle=':', color=colors[label])
    
    axs[1].set_xlabel('$\\ell$', fontsize=14)
    axs[1].set_ylabel(r'$\sigma(D_\ell^{BB}) / D_\ell^{BB}$', fontsize=14)
    axs[1].tick_params(axis='both', labelsize=12, direction="in", width=2, length=4)
    axs[1].tick_params(bottom=True, top=True, left=True, right=True, which = "major", direction="in", length=10, width=2);
    axs[1].tick_params(bottom=True, top=True, left=True, right=True, which = "minor", direction="in", length=5, width=1.5);
    axs[1].grid(which='major', linestyle='-', linewidth=0.5, alpha=0.3)
    for axis in ['top', 'bottom', 'left', 'right']:
        axs[1].spines[axis].set_linewidth(2)
    
    # axs[1].legend(fontsize=8)
    axs[1].set_xscale('log')
    axs[1].set_yscale('log')
    # axs[1].set_ylim(0,0.49)
    fig2.align_ylabels(axs)
    
    # Final layout
    plt.tight_layout()
    
    path_fig = Path('/afs/ihep.ac.cn/users/w/wangyiming25/tmp/20250726')
    path_fig.mkdir(exist_ok=True, parents=True)
    plt.savefig(path_fig / Path(f'res_std_nilc.png'), dpi=300)
    
    plt.show()


def get_ps_power():
    pass


def from_cl_to_bandpower():
    pass

plot_nilc()

