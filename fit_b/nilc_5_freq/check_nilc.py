import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pymaster as nmt

from pathlib import Path

beam = 17
lmax = 1500

rlz_idx = 0

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

def calc_dl_from_scalar_map(scalar_map, bl, apo_mask, bin_dl, masked_on_input):
    scalar_field = nmt.NmtField(apo_mask, [scalar_map], beam=bl, masked_on_input=masked_on_input, lmax=lmax, lmax_mask=lmax)
    dl = nmt.compute_full_master(scalar_field, scalar_field, bin_dl)
    return dl[0]

def calc_dl():
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax, pol=True)[:,2]
    l_min_edges, l_max_edges = generate_bins(l_min_start=30, delta_l_min=30, l_max=lmax+1, fold=0.2)
    # delta_ell = 30
    # bin_dl = nmt.NmtBin.from_nside_linear(nside, nlb=delta_ell, is_Dell=True)
    # bin_dl = nmt.NmtBin.from_lmax_linear(lmax=lmax, nlb=30, is_Dell=True)
    bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    ell_arr = bin_dl.get_effective_ells()

    apo_mask = np.load(f'../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5APO_3APO_5APO_3.npy')


    # m = np.load(f'./data/mean/pcfn/{rlz_idx}.npy')

    def _calc_dl(sim_mode, method):
        _map = np.load(f'./data/{sim_mode}/{method}/{rlz_idx}.npy')
        n_map = np.load(f'./data/{sim_mode}/n_{method}/{rlz_idx}.npy')

        dl = calc_dl_from_scalar_map(scalar_map=_map, bl=bl, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=False)
        dl_n = calc_dl_from_scalar_map(scalar_map=n_map, bl=bl, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=False)
        path_dl = Path(f'./dl_res/{sim_mode}/{method}')
        path_dl.mkdir(exist_ok=True, parents=True)
        path_dl_n = Path(f'./dl_res/{sim_mode}/n_{method}')
        path_dl_n.mkdir(exist_ok=True, parents=True)
        np.save(path_dl / Path(f'{rlz_idx}.npy'), dl)
        np.save(path_dl_n / Path(f'{rlz_idx}.npy'), dl_n)
        print(f'dl {sim_mode} {method} is ok')

    _calc_dl(sim_mode='mean', method='pcfn')
    _calc_dl(sim_mode='mean', method='cfn')
    _calc_dl(sim_mode='mean', method='rmv')
    _calc_dl(sim_mode='mean', method='inp')
    _calc_dl(sim_mode='std', method='pcfn')
    _calc_dl(sim_mode='std', method='cfn')
    _calc_dl(sim_mode='std', method='rmv')
    _calc_dl(sim_mode='std', method='inp')

calc_dl()

