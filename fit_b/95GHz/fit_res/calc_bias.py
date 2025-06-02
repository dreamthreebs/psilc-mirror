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
ps_mask = np.load(f'../inpainting/mask/apo_ps_mask.npy')

noise_seed = np.load('../../seeds_noise_2k.npy')
cmb_seed = np.load('../../seeds_cmb_2k.npy')
fg_seed = np.load('../../seeds_fg_2k.npy')


# utils
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

def calc_dl_correlation(m_q, m_u, m_q_2, m_u_2, bl, apo_mask, bin_dl, masked_on_input, purify_b):
    f2p = nmt.NmtField(apo_mask, [m_q, m_u], beam=bl, masked_on_input=masked_on_input, purify_b=purify_b, lmax=lmax, lmax_mask=lmax)
    f2p_2 = nmt.NmtField(apo_mask, [m_q_2, m_u_2], beam=bl, masked_on_input=masked_on_input, purify_b=purify_b, lmax=lmax, lmax_mask=lmax)
    w22p = nmt.NmtWorkspace.from_fields(f2p, f2p_2, bin_dl)
    # dl = nmt.workspaces.compute_full_master(pol_field, pol_field, b=bin_dl)
    dl = w22p.decouple_cell(nmt.compute_coupled_cell(f2p, f2p_2))[3]
    return dl

def calc_dl_from_scalar_map(scalar_map, bl, apo_mask, bin_dl, masked_on_input):
    scalar_field = nmt.NmtField(apo_mask, [scalar_map], beam=bl, masked_on_input=masked_on_input, lmax=lmax, lmax_mask=lmax)
    dl = nmt.compute_full_master(scalar_field, scalar_field, bin_dl)
    return dl[0]

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

def gen_test_map(rlz_idx=0, mode='mean', return_noise=False):
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

    cls = np.load('../../../src/cmbsim/cmbdata/cmbcl_8k.npy')
    if mode=='std':
        np.random.seed(seed=cmb_seed[rlz_idx])
    elif mode=='mean':
        np.random.seed(seed=cmb_seed[0])

    cmb_iqu = hp.synfast(cls.T, nside=nside, fwhm=np.deg2rad(beam)/60, new=True, lmax=3*nside-1)

    return cmb_iqu + noise, cmb_iqu, noise


# initialize the band power
bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax, pol=True)[:,2]
# l_min_edges, l_max_edges = generate_bins(l_min_start=10, delta_l_min=30, l_max=lmax+1, fold=0.2)
l_min_edges, l_max_edges = generate_bins(l_min_start=42, delta_l_min=40, l_max=lmax+1, fold=0.1, l_threshold=400)
# delta_ell = 30
# bin_dl = nmt.NmtBin.from_nside_linear(nside, nlb=delta_ell, is_Dell=True)
# bin_dl = nmt.NmtBin.from_lmax_linear(lmax=lmax, nlb=40, is_Dell=True)
bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
ell_arr = bin_dl.get_effective_ells()
print(f'{ell_arr=}')

# calc bias of 3 different methods
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

def bias_rmv():
    # calc bias from unresolved ps + model bias and individual terms. bias_all: rmv - cfn. bias_model: pcfn - rmv - resolved_ps bias_unresolved: unresolved_ps

    df = pd.read_csv(f"../mask/{freq}_after_filter.csv")
    flux_idx = 6
    lon = np.rad2deg(df.at[flux_idx, 'lon'])
    lat = np.rad2deg(df.at[flux_idx, 'lat'])

    ps_resolved = np.load(f'../data/ps/resolved_ps.npy')
    m_resolved_ps_q = ps_resolved[1].copy() * bin_mask
    m_resolved_ps_u = ps_resolved[2].copy() * bin_mask

    ps_unresolved = np.load(f'../data/ps/unresolved_ps.npy')
    m_unresolved_ps_q = ps_unresolved[1].copy() * bin_mask
    m_unresolved_ps_u = ps_unresolved[2].copy() * bin_mask

    # hp.orthview(m_resolved_ps_q, rot=[100,50,0], half_sky=True)
    # hp.orthview(m_resolved_ps_u, rot=[100,50,0], half_sky=True)
    # plt.show()
    # hp.orthview(m_unresolved_ps_q, rot=[100,50,0], half_sky=True)
    # hp.orthview(m_unresolved_ps_u, rot=[100,50,0], half_sky=True)
    # plt.show()

    m_pcfn, m_cfn, _, _= gen_map(rlz_idx=rlz_idx, mode='std')
    # np.save(f"./test_data/pcfn_{rlz_idx}.npy", m_pcfn)
    # np.save(f"./test_data/cfn_{rlz_idx}.npy", m_cfn)

    # m_pcfn = np.load(f"./test_data/pcfn_0.npy")
    # m_cfn = np.load(f"./test_data/cfn_0.npy")

    m_rmv_q = np.load(f'./std/3sigma/map_q_{rlz_idx}.npy') * bin_mask
    m_rmv_u = np.load(f'./std/3sigma/map_u_{rlz_idx}.npy') * bin_mask

    m_bias_model_q = m_pcfn[1].copy() * bin_mask - m_rmv_q - m_resolved_ps_q
    m_bias_model_u = m_pcfn[2].copy() * bin_mask - m_rmv_u - m_resolved_ps_u
    m_bias_all_q = m_rmv_q - m_cfn[1].copy() * bin_mask
    m_bias_all_u = m_rmv_u - m_cfn[2].copy() * bin_mask

    # hp.orthview(m_unresolved_ps_q, rot=[100,50,0], half_sky=True, title='bias unresolved ps')
    # hp.orthview(m_resolved_ps_q, rot=[100,50,0], half_sky=True, title='bias resolved ps')
    # hp.orthview(m_bias_all_q, rot=[100,50,0], half_sky=True, title='bias all q')
    # hp.orthview(m_bias_model_q, rot=[100,50,0], half_sky=True, title='bias model q')
    # plt.show()

    # hp.gnomview(m_unresolved_ps_q, rot=[lon,lat,0],  title='unresolved ps')
    # hp.gnomview(m_resolved_ps_q, rot=[lon,lat,0],  title='resolved ps')
    # hp.gnomview(m_bias_all_q, rot=[lon,lat,0],  title='bias all q')
    # hp.gnomview(m_bias_model_q, rot=[lon,lat,0],  title='bias model q')
    # plt.show()

    dl_bias_all = calc_dl_from_pol_map(m_q=m_bias_all_q, m_u=m_bias_all_u, bl=bl, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=False, purify_b=True)
    dl_bias_model = calc_dl_from_pol_map(m_q=m_bias_model_q, m_u=m_bias_model_u, bl=bl, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=False, purify_b=True)

    path_dl_qu_rmv = Path(f'BIAS/rmv')
    path_dl_qu_rmv.mkdir(parents=True, exist_ok=True)
    np.save(path_dl_qu_rmv / Path(f'bias_all_{rlz_idx}.npy'), dl_bias_all)
    np.save(path_dl_qu_rmv / Path(f'bias_model_{rlz_idx}.npy'), dl_bias_model)

def bias_inp():
    # calc bias from inpainting. bias all:inp - cfn
    # map is B mode in inpainting method

    # m_inp = hp.read_map(f"../inpainting/output_m3_std_new/{rlz_idx}.fits")
    m_cfn = hp.read_map(f"../inpainting/input_cfn_new/{rlz_idx}.fits")
    # m_bias_all = m_inp - m_cfn
    # hp.orthview(m_bias_all, rot=[100,50,0])
    # plt.show()

    # dl_bias_all = calc_dl_from_scalar_map(m_bias_all, bl, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=False)
    dl_cfn = calc_dl_from_scalar_map(m_cfn, bl, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=False)

    path_dl_qu_inp = Path(f'BIAS/inp')
    path_dl_qu_inp.mkdir(parents=True, exist_ok=True)
    np.save(path_dl_qu_inp/ Path(f'cfn_{rlz_idx}.npy'), dl_cfn)

def bias_mask():
    # calc bias after masking bias: pcfn - cfn
    ps = np.load(f'../../../fitdata/2048/PS/{freq}/ps.npy')

    dl_bias_all = calc_dl_from_pol_map(m_q=ps[1], m_u=ps[2], bl=bl, apo_mask=ps_mask, bin_dl=bin_dl, masked_on_input=False, purify_b=True)

    path_dl_qu_mask = Path(f'BIAS/mask')
    path_dl_qu_mask.mkdir(parents=True, exist_ok=True)
    np.save(path_dl_qu_mask/ Path(f'bias_all_{rlz_idx}.npy'), dl_bias_all)

def bias_rmv_from_correlation():
    # bias: after rmv - rmv residual map - 2 * rmv residual * cfn
    df = pd.read_csv(f"../mask/{freq}_after_filter.csv")
    flux_idx = 6
    lon = np.rad2deg(df.at[flux_idx, 'lon'])
    lat = np.rad2deg(df.at[flux_idx, 'lat'])

    m_pcfn, m_cfn, _, _= gen_map(rlz_idx=rlz_idx, mode='std')
    # np.save(f"./test_data/pcfn_{rlz_idx}.npy", m_pcfn)
    # np.save(f"./test_data/cfn_{rlz_idx}.npy", m_cfn)

    # m_pcfn = np.load(f"./test_data/pcfn_0.npy")
    # m_cfn = np.load(f"./test_data/cfn_0.npy")

    m_rmv_q = np.load(f'./std/3sigma/map_q_{rlz_idx}.npy') * bin_mask
    m_rmv_u = np.load(f'./std/3sigma/map_u_{rlz_idx}.npy') * bin_mask

    rmv_res_q = m_rmv_q - m_cfn[1] * bin_mask
    rmv_res_u = m_rmv_u - m_cfn[2] * bin_mask

    dl_rmv = calc_dl_from_pol_map(m_q=m_rmv_q, m_u=m_rmv_u, bl=bl, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=False, purify_b=True)
    dl_cor = calc_dl_correlation(m_q=rmv_res_q, m_u=rmv_res_u, m_q_2=m_cfn[1], m_u_2=m_cfn[2], bl=bl, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=False, purify_b=True)

    path_dl_qu_rmv = Path(f'BIAS/rmv_cor')
    path_dl_qu_rmv.mkdir(parents=True, exist_ok=True)
    np.save(path_dl_qu_rmv / Path(f'rmv_{rlz_idx}.npy'), dl_rmv)
    np.save(path_dl_qu_rmv / Path(f'cor_{rlz_idx}.npy'), dl_cor)


# test calc correlation between point sources and other components
def bias_correlation():
    ps = np.load(f'../../../fitdata/2048/PS/{freq}/ps.npy')
    m_pcfn, m_cfn, _, _= gen_map(rlz_idx=rlz_idx, mode='std')

    dl_ps_cfn_cor = calc_dl_correlation(m_q=ps[1], m_u=ps[2],m_q_2=m_cfn[1], m_u_2=m_cfn[2], bl=bl, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=False, purify_b=True)
    path_dl_qu_mask = Path(f'BIAS/test_correlation')
    path_dl_qu_mask.mkdir(parents=True, exist_ok=True)
    np.save(path_dl_qu_mask/ Path(f'ps_cfn_{rlz_idx}.npy'), dl_ps_cfn_cor)

def test_correlation_cmb_noise():
    cn, c, n = gen_test_map(mode='std')

    dl_c = calc_dl_from_pol_map(m_q=c[1], m_u=c[2], bl=bl, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=False, purify_b=True)
    dl_n = calc_dl_from_pol_map(m_q=n[1], m_u=n[2], bl=bl, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=False, purify_b=True)
    dl_cn = calc_dl_from_pol_map(m_q=cn[1], m_u=cn[2], bl=bl, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=False, purify_b=True)
    dl_cn_cor = calc_dl_correlation(m_q=c[1], m_u=c[2], m_q_2=n[1], m_u_2=n[2], bl=bl, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=False, purify_b=True)
    path_dl_qu_mask = Path(f'BIAS/test_correlation_cmb_noise')
    path_dl_qu_mask.mkdir(parents=True, exist_ok=True)
    np.save(path_dl_qu_mask/ Path(f'c_{rlz_idx}.npy'), dl_c)
    np.save(path_dl_qu_mask/ Path(f'n_{rlz_idx}.npy'), dl_n)
    np.save(path_dl_qu_mask/ Path(f'cn_{rlz_idx}.npy'), dl_cn)
    np.save(path_dl_qu_mask/ Path(f'cn_cor_{rlz_idx}.npy'), dl_cn_cor)




if __name__ == "__main__":
    # bias_pcfn()
    # bias_unresolved()
    # bias_rmv()
    bias_inp()
    # bias_mask()
    # bias_correlation()
    # bias_rmv_from_correlation()
