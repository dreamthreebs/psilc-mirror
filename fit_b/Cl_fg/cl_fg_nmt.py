import numpy as np
import healpy as hp
import pymaster as nmt
import matplotlib.pyplot as plt

from pathlib import Path

lmax = 600
nside = 2048
beam = 67
fg = np.load('../../fitdata/2048/FG/30/fg.npy')
apo_mask = np.load('../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5APO_5.npy')
bin_mask = np.load('../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5.npy')

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
    pol_field = nmt.NmtField(apo_mask, [m_q, m_u], beam=bl, masked_on_input=masked_on_input, purify_b=purify_b)
    dl = nmt.compute_full_master(pol_field, pol_field, bin_dl)
    return dl[3]

def calc_dl_from_scalar_map(scalar_map, bl, apo_mask, bin_dl, masked_on_input):
    scalar_field = nmt.NmtField(apo_mask, [scalar_map], beam=bl, masked_on_input=masked_on_input)
    dl = nmt.compute_full_master(scalar_field, scalar_field, bin_dl)
    return dl[0]

def get_dl_pol():
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=7000, pol=True)[:,2]
    l_min_edges, l_max_edges = generate_bins(l_min_start=30, delta_l_min=20, l_max=lmax, fold=0.2)
    bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    ell_arr = bin_dl.get_effective_ells()
    
    dl_fg = calc_dl_from_pol_map(m_q=fg[1], m_u=fg[2], bl=bl, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=False, purify_b=True)
    
    dl_data_path = Path('nmt_data/pol')
    dl_data_path.mkdir(exist_ok=True, parents=True)
    np.save(dl_data_path / Path('dl_fg.npy'), dl_fg)


def get_dl_sca():
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=7000, pol=True)[:,2]
    l_min_edges, l_max_edges = generate_bins(l_min_start=30, delta_l_min=20, l_max=lmax, fold=0.2)
    bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    ell_arr = bin_dl.get_effective_ells()

    m_fg = hp.alm2map(hp.map2alm(fg)[2], nside=nside)

    dl_fg = calc_dl_from_scalar_map(scalar_map=m_fg, bl=bl, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=False)
    
    dl_data_path = Path('nmt_data/sca')
    dl_data_path.mkdir(exist_ok=True, parents=True)
    np.save(dl_data_path / Path('dl_fg.npy'), dl_fg)

# get_dl_sca()

def check_fg_dl():
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=7000, pol=True)[:,2]
    l_min_edges, l_max_edges = generate_bins(l_min_start=30, delta_l_min=20, l_max=lmax, fold=0.2)
    bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    ell_arr = bin_dl.get_effective_ells()

    l = np.arange(lmax+1)
    cl_hp = np.load('./data_full_lmax3nside/cl_fg_BB.npy')

    dl_pol = np.load('./nmt_data/pol/dl_fg.npy')
    dl_sca = np.load('./nmt_data/sca/dl_fg.npy')

    plt.loglog(ell_arr, dl_pol, label='pol')
    plt.loglog(ell_arr, dl_sca, label='sca')
    plt.loglog(l, l*(l+1)*cl_hp/bl[:lmax+1]**2/(2*np.pi), label='healpy')
    plt.legend()
    plt.show()


check_fg_dl()
