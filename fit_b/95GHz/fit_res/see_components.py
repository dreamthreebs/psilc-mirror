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

bin_mask = np.load('../../../src/mask/north/BINMASKG2048.npy')
apo_mask = np.load('../../../psfit/fitv4/fit_res/2048/ps_mask/new_mask/apo_C1_3_apo_3_apo_3.npy')
# ps_mask = np.load(f'../inpainting/mask/apo_ps_mask.npy')

noise_seeds = np.load('../../seeds_noise_2k.npy')
cmb_seeds = np.load('../../seeds_cmb_2k.npy')
fg_seeds = np.load('../../seeds_fg_2k.npy')

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
    dl = w22p.decouple_cell(nmt.compute_coupled_cell(f2p, f2p))
    # return EE, BB
    return dl[0], dl[3]

def gen_fg_cl():
    cl_fg = np.load('../data/debeam_full_b/cl_fg.npy')
    Cl_TT = cl_fg[0]
    Cl_EE = cl_fg[1]
    Cl_BB = cl_fg[2]
    Cl_TE = np.zeros_like(Cl_TT)
    return np.array([Cl_TT, Cl_EE, Cl_BB, Cl_TE])

def gen_map(rlz_idx=0, mode='mean', return_noise=False):
    # mode can be mean or std
    noise_seed = np.load('../../seeds_noise_2k.npy')
    cmb_seed = np.load('../../seeds_cmb_2k.npy')
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
    return pcfn, cfn, cf, n, cmb_iqu, fg, ps


def cpr_spectrum_pcn_b(bin_mask, apo_mask):

    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax, pol=True)[:,2]
    # l_min_edges, l_max_edges = generate_bins(l_min_start=10, delta_l_min=30, l_max=lmax+1, fold=0.2)
    l_min_edges, l_max_edges = generate_bins(l_min_start=42, delta_l_min=40, l_max=lmax+1, fold=0.1, l_threshold=400)
    # delta_ell = 30
    # bin_dl = nmt.NmtBin.from_nside_linear(nside, nlb=delta_ell, is_Dell=True)
    # bin_dl = nmt.NmtBin.from_lmax_linear(lmax=lmax, nlb=40, is_Dell=True)
    bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    ell_arr = bin_dl.get_effective_ells()

    # m_c = np.load(f'../../../../fitdata/2048/CMB/{freq}/{rlz_idx}.npy')
    # m_cn = np.load(f'../../../../fitdata/synthesis_data/2048/CMBNOISE/{freq}/{rlz_idx}.npy')
    # m_pcn = np.load(f'../../../../fitdata/synthesis_data/2048/PSCMBNOISE/{freq}/{rlz_idx}.npy')

    m_pcfn, m_cfn, m_cf, m_n, m_c, m_f, m_p = gen_map(rlz_idx=rlz_idx, mode='std')

    # m_pcfn_q = np.load(f'./pcfn_fit_qu/3sigma/map_q_{rlz_idx}.npy') * bin_mask
    # m_pcfn_u = np.load(f'./pcfn_fit_qu/3sigma/map_u_{rlz_idx}.npy') * bin_mask

    # m_n_q = np.load(f'./pcfn_fit_qu_n/3sigma/map_q_{rlz_idx}.npy') * bin_mask
    # m_n_u = np.load(f'./pcfn_fit_qu_n/3sigma/map_u_{rlz_idx}.npy') * bin_mask

    m_pcfn_q = m_pcfn[1].copy() * bin_mask
    m_pcfn_u = m_pcfn[2].copy() * bin_mask

    m_cfn_q = m_cfn[1].copy() * bin_mask
    m_cfn_u = m_cfn[2].copy() * bin_mask

    m_cf_q = m_cf[1].copy() * bin_mask
    m_cf_u = m_cf[2].copy() * bin_mask

    m_n_q = m_n[1].copy() * bin_mask
    m_n_u = m_n[2].copy() * bin_mask

    m_c_q = m_c[1].copy() * bin_mask
    m_c_u = m_c[2].copy() * bin_mask

    m_p_q = m_p[1].copy() * bin_mask
    m_p_u = m_p[2].copy() * bin_mask

    m_f_q = m_f[1].copy() * bin_mask
    m_f_u = m_f[2].copy() * bin_mask

    print('begin calc dl...')

    dl_pcfn = calc_dl_from_pol_map(m_q=m_pcfn_q, m_u=m_pcfn_u, bl=bl, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=False, purify_b=True)
    dl_cfn = calc_dl_from_pol_map(m_q=m_cfn_q, m_u=m_cfn_u, bl=bl, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=False, purify_b=True)
    dl_cf = calc_dl_from_pol_map(m_q=m_cf_q, m_u=m_cf_u, bl=bl, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=False, purify_b=True)
    dl_n = calc_dl_from_pol_map(m_q=m_n_q, m_u=m_n_u, bl=bl, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=False, purify_b=True)
    dl_c = calc_dl_from_pol_map(m_q=m_c_q, m_u=m_c_u, bl=bl, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=False, purify_b=True)
    dl_f = calc_dl_from_pol_map(m_q=m_f_q, m_u=m_f_u, bl=bl, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=False, purify_b=True)
    dl_p = calc_dl_from_pol_map(m_q=m_p_q, m_u=m_p_u, bl=bl, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=False, purify_b=True)

    path_dl_qu_pcfn = Path(f'pcfn_dl4/component/pcfn')
    path_dl_qu_cfn = Path(f'pcfn_dl4/component/cfn')
    path_dl_qu_cf = Path(f'pcfn_dl4/component/cf')
    path_dl_qu_n = Path(f'pcfn_dl4/component/n')
    path_dl_qu_p = Path(f'pcfn_dl4/component/p')
    path_dl_qu_f = Path(f'pcfn_dl4/component/f')
    path_dl_qu_c = Path(f'pcfn_dl4/component/c')
    path_dl_qu_pcfn.mkdir(parents=True, exist_ok=True)
    path_dl_qu_cfn.mkdir(parents=True, exist_ok=True)
    path_dl_qu_cf.mkdir(parents=True, exist_ok=True)
    path_dl_qu_n.mkdir(parents=True, exist_ok=True)
    path_dl_qu_p.mkdir(parents=True, exist_ok=True)
    path_dl_qu_c.mkdir(parents=True, exist_ok=True)
    path_dl_qu_f.mkdir(parents=True, exist_ok=True)

    np.save(path_dl_qu_pcfn / Path(f'{rlz_idx}.npy'), dl_pcfn)
    np.save(path_dl_qu_cfn / Path(f'{rlz_idx}.npy'), dl_cfn)
    np.save(path_dl_qu_cf / Path(f'{rlz_idx}.npy'), dl_cf)
    np.save(path_dl_qu_n / Path(f'{rlz_idx}.npy'), dl_n)
    np.save(path_dl_qu_p / Path(f'{rlz_idx}.npy'), dl_p)
    np.save(path_dl_qu_f / Path(f'{rlz_idx}.npy'), dl_f)
    np.save(path_dl_qu_c / Path(f'{rlz_idx}.npy'), dl_c)

def plot_power():
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax, pol=True)[:,2]
    # l_min_edges, l_max_edges = generate_bins(l_min_start=10, delta_l_min=30, l_max=lmax+1, fold=0.2)
    l_min_edges, l_max_edges = generate_bins(l_min_start=42, delta_l_min=40, l_max=lmax+1, fold=0.1, l_threshold=400)
    # delta_ell = 30
    # bin_dl = nmt.NmtBin.from_nside_linear(nside, nlb=delta_ell, is_Dell=True)
    # bin_dl = nmt.NmtBin.from_lmax_linear(lmax=lmax, nlb=40, is_Dell=True)
    bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    ell_arr = bin_dl.get_effective_ells()
    l_start = int(ell_arr[0])
    cls = np.load('../../../src/cmbsim/cmbdata/cmbcl_8k.npy').T[:,l_start:lmax+1]
    l = np.arange(lmax+1)[l_start:lmax+1]

    pcfn_mean = np.mean([np.load(f'./pcfn_dl4/component/pcfn/{rlz_idx}.npy')[0] - np.load(f'./pcfn_dl4/component/n/{rlz_idx}.npy')[0] for rlz_idx in np.arange(1,200)], axis=0)
    cfn_mean = np.mean([np.load(f'./pcfn_dl4/component/cfn/{rlz_idx}.npy')[0] - np.load(f'./pcfn_dl4/component/n/{rlz_idx}.npy')[0] for rlz_idx in np.arange(1,200)], axis=0)
    p_mean = np.mean([np.load(f'./pcfn_dl4/component/p/{rlz_idx}.npy')[0] for rlz_idx in np.arange(1,200)], axis=0)
    f_mean = np.mean([np.load(f'./pcfn_dl4/component/f/{rlz_idx}.npy')[0] for rlz_idx in np.arange(1,200)], axis=0)
    n_mean = np.mean([np.load(f'./pcfn_dl4/component/n/{rlz_idx}.npy')[0] for rlz_idx in np.arange(1,200)], axis=0)


    plt.loglog(ell_arr, pcfn_mean, label='ps+cmb+fg+noise')
    plt.loglog(ell_arr, cfn_mean, label='cmb+fg+noise')
    plt.loglog(ell_arr, p_mean, label='ps')
    plt.loglog(ell_arr, f_mean, label='fg')
    plt.loglog(ell_arr, n_mean, label='noise')
    plt.loglog(l, l*(l+1)*cls[1]/(2*np.pi), label='fiducial CMB', color='black')
    plt.legend()
    plt.title("different component comparison")
    plt.show()



def main():

    # cpr_spectrum_pcn_b(bin_mask=bin_mask, apo_mask=apo_mask)
    plot_power()

main()











