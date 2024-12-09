import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pymaster as nmt
import pandas as pd

from pathlib import Path
from config import lmax, nside, beam, freq
from scipy.interpolate import interp1d
from pix_cov_qu import CovCalculator
from fit_qu_no_const import FitPolPS

df = pd.read_csv(f'./mask/{freq}.csv')
df_info = pd.read_csv(f'../../FGSim/FreqBand')
ori_mask = np.load('../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5.npy')
ori_apo_mask = np.load('../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5APO_5.npy')

noise_seeds = np.load('../seeds_noise_2k.npy')
cmb_seeds = np.load('../seeds_cmb_2k.npy')
fg_seeds = np.load('../seeds_fg_2k.npy')
rlz_idx = 0

# Part 1: Utils
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
    f2p = nmt.NmtField(apo_mask, [m_q, m_u], beam=bl, masked_on_input=masked_on_input, purify_b=purify_b, lmax=lmax, lmax_mask=lmax)
    w22p = nmt.NmtWorkspace.from_fields(f2p, f2p, bin_dl)
    # dl = nmt.workspaces.compute_full_master(pol_field, pol_field, b=bin_dl)
    dl = w22p.decouple_cell(nmt.compute_coupled_cell(f2p, f2p)) # dl[0] for EE, dl[3] for BB
    return dl

def gen_cmb_cl(beam, lmax):
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=10000, pol=True)
    print(f'{bl[0:10,0]=}')
    print(f'{bl[0:10,1]=}')
    print(f'{bl[0:10,2]=}')
    print(f'{bl[0:10,3]=}')
    # cl = np.load('../../src/cmbsim/cmbdata/cmbcl.npy')
    cl = np.load('../../src/cmbsim/cmbdata/cmbcl_8k.npy')
    print(f'{cl.shape=}')

    Cl_TT = cl[0:lmax+1,0] * bl[0:lmax+1,0]**2
    Cl_EE = cl[0:lmax+1,1] * bl[0:lmax+1,1]**2
    Cl_BB = cl[0:lmax+1,2] * bl[0:lmax+1,2]**2
    Cl_TE = cl[0:lmax+1,3] * bl[0:lmax+1,3]**2
    return np.asarray([Cl_TT, Cl_EE, Cl_BB, Cl_TE])

def gen_ps_fg_map():
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax, pol=True)[:,2]
    l_min_edges, l_max_edges = generate_bins(l_min_start=30, delta_l_min=30, l_max=lmax+1, fold=0.2)
    # delta_ell = 30
    # bin_dl = nmt.NmtBin.from_nside_linear(nside, nlb=delta_ell, is_Dell=True)
    # bin_dl = nmt.NmtBin.from_lmax_linear(lmax=lmax, nlb=30, is_Dell=True)
    bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)

    ell_arr = bin_dl.get_effective_ells()

    ps = np.load(f'../../fitdata/2048/PS/{freq}/ps.npy')
    fg = np.load(f'../../fitdata/2048/FG/{freq}/fg.npy')

    pf = ps + fg

    dl_pf = calc_dl_from_pol_map(m_q=pf[1], m_u=pf[2], bl=bl, apo_mask=ori_apo_mask, bin_dl=bin_dl, masked_on_input=False, purify_b=True)
    dl_f = calc_dl_from_pol_map(m_q=fg[1], m_u=fg[2], bl=bl, apo_mask=ori_apo_mask, bin_dl=bin_dl, masked_on_input=False, purify_b=True)
    dl_p = dl_pf - dl_f

    cmb_cl = gen_cmb_cl(beam=0, lmax=lmax)
    l = np.arange(lmax+1)
    dl_c = l * (l+1) * cmb_cl / (2 * np.pi)

    map_depth = df_info.at[3, 'mapdepth']
    nl = (map_depth/bl)**2 / 3437.728**2
    dl_n = l * (l+1) * nl / (2 * np.pi)

    path_check_component = Path('./component_dl')
    path_check_component.mkdir(parents=True, exist_ok=True)
    np.save(path_check_component / 'ell_arr.npy', ell_arr)
    np.save(path_check_component / 'dl_p.npy', dl_p)
    np.save(path_check_component / 'dl_f.npy', dl_f)
    np.save(path_check_component / 'dl_c.npy', dl_c)
    np.save(path_check_component / 'dl_n.npy', dl_n)


def check_component():

    dl_c = np.load('./component_dl/dl_c.npy')
    dl_p = np.load('./component_dl/dl_p.npy')
    dl_f = np.load('./component_dl/dl_f.npy')
    dl_n = np.load('./component_dl/dl_n.npy')
    ell_arr = np.load('./component_dl/ell_arr.npy')
    l = np.arange(lmax+1)

    plt.loglog(ell_arr, dl_p[0], label='point source')
    plt.loglog(ell_arr, dl_f[0], label='diffuse foreground')
    plt.loglog(l, dl_c[1], label='cmb')
    plt.loglog(l, dl_n, label='noise')
    plt.legend()
    # plt.savefig(f'/afs/ihep.ac.cn/users/w/wangyiming25/tmp/20241209/{freq}.png', dpi=300)
    plt.show()

check_component()


# gen_ps_fg_map()



