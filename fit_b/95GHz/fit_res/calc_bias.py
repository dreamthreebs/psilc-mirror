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

rlz_idx=0
threshold = 3

df = pd.read_csv('../../../FGSim/FreqBand')
print(f'{freq=}, {beam=}')

# bin_mask = np.load('../../../src/mask/north/BINMASKG2048.npy')
bin_mask = np.load('../../../psfit/fitv4/fit_res/2048/ps_mask/new_mask/BIN_C1_3_C1_3.npy')
apo_mask = np.load('../../../psfit/fitv4/fit_res/2048/ps_mask/new_mask/apo_C1_3_apo_3_apo_3.npy')
print(f'{np.sum(apo_mask)/np.size(apo_mask)=}')
# ps_mask = np.load(f'../inpainting/mask/apo_ps_mask.npy')

noise_seed = np.load('../../seeds_noise_2k.npy')
cmb_seed = np.load('../../seeds_cmb_2k.npy')
fg_seed = np.load('../../seeds_fg_2k.npy')

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

def calc_dl_from_pol_map(m_q, m_u, bl, apo_mask, bin_dl, masked_on_input, purify_b):
    f2p = nmt.NmtField(apo_mask, [m_q, m_u], beam=bl, masked_on_input=masked_on_input, purify_b=purify_b, lmax=lmax, lmax_mask=lmax)
    w22p = nmt.NmtWorkspace.from_fields(f2p, f2p, bin_dl)
    # dl = nmt.workspaces.compute_full_master(pol_field, pol_field, b=bin_dl)
    dl = w22p.decouple_cell(nmt.compute_coupled_cell(f2p, f2p))[3]
    return dl

def gen_fg_cl():
    cl_fg = np.load('../data/debeam_full_b/cl_fg.npy')
    Cl_TT = cl_fg[0]
    Cl_EE = cl_fg[1]
    Cl_BB = cl_fg[2]
    Cl_TE = np.zeros_like(Cl_TT)
    return np.array([Cl_TT, Cl_EE, Cl_BB, Cl_TE])

def gen_map(rlz_idx=0, mode='mean', return_noise=False):
    # mode can be mean or std
    nside = 2048

    nstd = np.load(f'../../../FGSim/NSTDNORTH/2048/{freq}.npy')
    npix = hp.nside2npix(nside=2048)
    np.random.seed(seed=noise_seed[rlz_idx])
    # noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))
    noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))
    print(f"{np.std(noise[1])=}")

    if return_noise:
        return noise

    ps = np.load(f'../../../fitdata/2048/PS/{freq}/ps.npy')
    fg = np.load(f'../../../fitdata/2048/FG/{freq}/fg.npy')

    cls = np.load('../../../src/cmbsim/cmbdata/cmbcl_8k.npy')
    if mode=='std':
        np.random.seed(seed=cmb_seed[rlz_idx])
    elif mode=='mean':
        np.random.seed(seed=cmb_seed[0])

    cmb_iqu = hp.synfast(cls.T, nside=nside, fwhm=np.deg2rad(beam)/60, new=True, lmax=3*nside-1)

    pcfn = noise + ps + cmb_iqu + fg
    cfn = noise + cmb_iqu + fg
    cf = cmb_iqu + fg
    n = noise
    return pcfn, cfn, cf, n

def bias_pcfn():
    ps = np.load(f'../../../fitdata/2048/PS/{freq}/ps.npy')

    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax, pol=True)[:,2]
    # l_min_edges, l_max_edges = generate_bins(l_min_start=10, delta_l_min=30, l_max=lmax+1, fold=0.2)
    l_min_edges, l_max_edges = generate_bins(l_min_start=42, delta_l_min=40, l_max=lmax+1, fold=0.1, l_threshold=400)
    # delta_ell = 30
    # bin_dl = nmt.NmtBin.from_nside_linear(nside, nlb=delta_ell, is_Dell=True)
    # bin_dl = nmt.NmtBin.from_lmax_linear(lmax=lmax, nlb=40, is_Dell=True)
    bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    ell_arr = bin_dl.get_effective_ells()
    print(f'{ell_arr=}')

    m_ps_q = ps[1].copy() * bin_mask
    m_ps_u = ps[2].copy() * bin_mask

    # hp.orthview(m_ps_q, rot=[100,50,0], half_sky=True)
    # hp.orthview(m_ps_u, rot=[100,50,0], half_sky=True)
    # plt.show()

    print("Begin calc m_ps's power!")
    dl_ps = calc_dl_from_pol_map(m_q=m_ps_q, m_u=m_ps_u, bl=bl, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=False, purify_b=True)
    print("Done calc m_ps's power!")

    path_dl_qu_pcfn = Path(f'BIAS/pcfn')
    path_dl_qu_pcfn.mkdir(parents=True, exist_ok=True)

    np.save(path_dl_qu_pcfn / Path(f'{rlz_idx}.npy'), dl_ps)

def bias_unresolved():
    ps = np.load(f'../data/ps/unresolved_ps.npy')

    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax, pol=True)[:,2]
    # l_min_edges, l_max_edges = generate_bins(l_min_start=10, delta_l_min=30, l_max=lmax+1, fold=0.2)
    l_min_edges, l_max_edges = generate_bins(l_min_start=42, delta_l_min=40, l_max=lmax+1, fold=0.1, l_threshold=400)
    # delta_ell = 30
    # bin_dl = nmt.NmtBin.from_nside_linear(nside, nlb=delta_ell, is_Dell=True)
    # bin_dl = nmt.NmtBin.from_lmax_linear(lmax=lmax, nlb=40, is_Dell=True)
    bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    ell_arr = bin_dl.get_effective_ells()
    print(f'{ell_arr=}')

    m_ps_q = ps[1].copy() * bin_mask
    m_ps_u = ps[2].copy() * bin_mask

    # hp.orthview(m_ps_q, rot=[100,50,0], half_sky=True)
    # hp.orthview(m_ps_u, rot=[100,50,0], half_sky=True)
    # plt.show()

    print("Begin calc m_ps's power!")
    dl_ps = calc_dl_from_pol_map(m_q=m_ps_q, m_u=m_ps_u, bl=bl, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=False, purify_b=True)
    print("Done calc m_ps's power!")

    path_dl_qu_pcfn = Path(f'BIAS/unresolved_ps')
    path_dl_qu_pcfn.mkdir(parents=True, exist_ok=True)

    np.save(path_dl_qu_pcfn / Path(f'{rlz_idx}.npy'), dl_ps)




if __name__ == "__main__":
    # bias_pcfn()
    bias_unresolved()
