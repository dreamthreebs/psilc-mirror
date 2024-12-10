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

def dl2cl(D_ell):

    ell = np.arange(len(D_ell))
    mask = ell > 1
    C_ell = np.zeros_like(D_ell, dtype=np.float64)
    C_ell[mask] = (2 * np.pi * D_ell[mask]) / (ell[mask] * (ell[mask] + 1))
    C_ell[~mask] = 0
    return C_ell

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

    map_depth = df_info.at[6, 'mapdepth']
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

# Part 2: estimate power spectrum of Cl total
def gen_fg_cl():
    cl_fg = np.load('./data/debeam_full_b/cl_fg.npy')
    Cl_TT = cl_fg[0]
    Cl_EE = cl_fg[1]
    Cl_BB = cl_fg[2]
    Cl_TE = np.zeros_like(Cl_TT)
    return np.array([Cl_TT, Cl_EE, Cl_BB, Cl_TE])

def plot_th_cf(map_type, with_beam=True):
    # Dl theoretical cmb + foreground

    # cfn = np.load(f'./cfn.npy')
    # cf = np.load(f'./cf.npy')
    # n = np.load(f'./n.npy')
    l = np.arange(lmax+1)
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax, pol=True).T
    cl_th_fg = gen_fg_cl()
    cl_th_cmb = gen_cmb_cl(beam=beam, lmax=lmax)

    if map_type == "EE":
        i = 1
    elif map_type == "BB":
        i = 2

    if with_beam:
        # plt.loglog(l*(l+1)*cl_th_fg[i]/bl[i]**2/(2*np.pi), label=f'fg, {i=}')
        # plt.loglog(l*(l+1)*cl_th_cmb[i]/bl[i]**2/(2*np.pi), label=f'cmb, {i=}')
        plt.loglog(l*(l+1)*(cl_th_cmb[i]+cl_th_fg[i])/bl[i]**2/(2*np.pi), label=f'cmb+fg input {map_type}')
        # plt.loglog(l*(l+1)*(cl_th_cmb[i]+cl_th_fg[i])/(2*np.pi), label=f'cmb+fg no beam {map_type}')
    else:
        # plt.loglog(l*(l+1)*cl_th_fg[i]/(2*np.pi), label=f'fg, {i=}')
        # plt.loglog(l*(l+1)*cl_th_cmb[i]/(2*np.pi), label=f'cmb, {i=}')
        plt.loglog(l*(l+1)*(cl_th_cmb[i]+cl_th_fg[i])/(2*np.pi), label=f'cmb+fg input {map_type}')

def gen_map(rlz_idx):
    npix = hp.nside2npix(nside=nside)
    ps = np.load('./data/ps/ps.npy')

    nstd = np.load(f'../../FGSim/NSTDNORTH/2048/{freq}.npy')
    np.random.seed(seed=noise_seeds[rlz_idx])
    # noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))
    noise = nstd * np.random.normal(loc=0, scale=1, size=(3,npix))
    print(f"{np.std(noise[1])=}")

    # cmb_iqu = np.load(f'../../fitdata/2048/CMB/215/{rlz_idx}.npy')
    # cls = np.load('../../src/cmbsim/cmbdata/cmbcl.npy')
    cls = np.load('../../src/cmbsim/cmbdata/cmbcl_8k.npy')
    np.random.seed(seed=cmb_seeds[rlz_idx])
    # cmb_iqu = hp.synfast(cls.T, nside=nside, fwhm=np.deg2rad(beam)/60, new=True, lmax=1999)
    cmb_iqu = hp.synfast(cls.T, nside=nside, fwhm=np.deg2rad(beam)/60, new=True, lmax=lmax)

    cls_fg = gen_fg_cl()
    np.random.seed(seed=fg_seeds[rlz_idx])
    fg = hp.synfast(cls_fg, nside=nside, fwhm=0, new=True, lmax=lmax)

    pcfn = noise + ps + cmb_iqu + fg
    cfn = noise + cmb_iqu + fg
    cf = cmb_iqu + fg
    n = noise

    # pcfn = noise
    # cfn = noise
    # cf = noise
    # n = noise

    return pcfn, cfn, cf, n

def calc_th_cov():
    cl_fg = gen_fg_cl()
    cl_cmb = gen_cmb_cl(beam=beam, lmax=lmax)

    cl_tot = cl_fg + cl_cmb

    flux_idx = 0
    pix_ind = np.load(f'./pix_idx_qu/{flux_idx}.npy')
    obj_cov = CovCalculator(nside=nside, lmin=2, lmax=lmax, Cl_TT=cl_tot[0], Cl_EE=cl_tot[1], Cl_BB=cl_tot[2], Cl_TE=cl_tot[3], pixind=pix_ind, calc_opt='polarization', out_pol_opt='QU')
    MP = obj_cov.run_calc_cov()

    path_cov = Path('./cmb_qu_cov')
    path_cov.mkdir(exist_ok=True, parents=True)
    np.save(Path(path_cov / f'{flux_idx}.npy'), MP)

def do_th_fit():
    nstd = np.load(f'../../FGSim/NSTDNORTH/2048/{freq}.npy')
    df_mask = pd.read_csv(f'./mask/{freq}.csv')
    pcfn,_,_,_ = gen_map(rlz_idx=rlz_idx)

    obj = FitPolPS(m_q=pcfn[1], m_u=pcfn[2], freq=freq, nstd_q=nstd[1], nstd_u=nstd[2], flux_idx=0, df_mask=df_mask, df_ps=df_mask, lmax=lmax, nside=nside, radius_factor=1.5, beam=beam)

    # obj.calc_definite_fixed_cmb_cov()
    # obj.calc_covariance_matrix(mode='cmb+noise')
    num_ps, chi2dof, fit_P, fit_P_err, fit_phi, fit_phi_err = obj.fit_all(cov_mode='cmb+noise')
    path_res = Path('./parameter/th_have_all')
    path_res.mkdir(exist_ok=True, parents=True)
    np.save(path_res / Path(f'fit_P_{rlz_idx}.npy'), fit_P)
    np.save(path_res / Path(f'fit_phi_{rlz_idx}.npy'), fit_phi)

# Part 3: Interpolate dl
def interp_band_power(ell_arr, band_power, lmax_cov, lmax_interp=None):
    # from band power dl to dl on all ell, interp on log scale, and only use data up to lmax_interp
    print(f'before interp, {band_power=} ')
    l = np.arange(lmax_cov + 1)
    band_power[band_power < 1e-10] = 1e-10
    log_band_power = np.log(band_power)

    if lmax_interp is None:
        lmax_interp = lmax_cov + 200

    idx_need = np.searchsorted(ell_arr, lmax_interp, side='right') - 1
    print(f'{lmax_cov=}, {ell_arr[idx_need]=}')

    interp_func = interp1d(ell_arr[:idx_need], log_band_power[:idx_need], kind='cubic', fill_value='extrapolate')
    log_dl_interp = interp_func(l)
    dl_interp = np.exp(log_dl_interp)
    dl_interp[l<ell_arr[0]] = band_power[0]
    return l, dl_interp

def interp_dl(lmax_cov, lmax_interp=None):

    # pcfn, cfn, cf, n = gen_map(rlz_idx=rlz_idx)
    pcfn = np.load('./component_dl/one_rlz_map/pcfn.npy')
    cfn = np.load('./component_dl/one_rlz_map/cfn.npy')
    cf = np.load('./component_dl/one_rlz_map/cf.npy')
    n = np.load('./component_dl/one_rlz_map/n.npy')

    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax, pol=True)[:,2]
    l_min_edges, l_max_edges = generate_bins(l_min_start=20, delta_l_min=10, l_max=lmax+1, fold=0.1)
    bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    ell_arr = bin_dl.get_effective_ells()

    dl_cfn = calc_dl_from_pol_map(m_q=pcfn[1].copy(), m_u=pcfn[2].copy(), bl=bl, apo_mask=ori_apo_mask, bin_dl=bin_dl, masked_on_input=False, purify_b=True)
    dl_n = calc_dl_from_pol_map(m_q=n[1].copy(), m_u=n[2].copy(), bl=bl, apo_mask=ori_apo_mask, bin_dl=bin_dl, masked_on_input=False, purify_b=True)

    l, dl_interp_e = interp_band_power(ell_arr=ell_arr, band_power=dl_cfn[0]-dl_n[0], lmax_cov=lmax_cov, lmax_interp=lmax_interp)
    l, dl_interp_b = interp_band_power(ell_arr=ell_arr, band_power=dl_cfn[3]-dl_n[3], lmax_cov=lmax_cov, lmax_interp=lmax_interp)
    # plot_th_cf(map_type='EE')
    cl_interp_e = dl2cl(dl_interp_e)
    cl_interp_b = dl2cl(dl_interp_b)

    path_cl_interp = Path('./component_dl/cl_interp_pcfn')
    path_cl_interp.mkdir(exist_ok=True, parents=True)
    np.save(path_cl_interp / Path(f'e.npy'), cl_interp_e)
    np.save(path_cl_interp / Path(f'b.npy'), cl_interp_b)

def check_interp_cl():
    cl_interp_e = np.load('./component_dl/cl_interp_pcfn/e.npy')
    cl_interp_b = np.load('./component_dl/cl_interp_pcfn/b.npy')
    print(f'{cl_interp_b=}')
    l = np.arange(np.size(cl_interp_e))
    # print(f'{l.shape=}')
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=np.size(cl_interp_e)-1)

    plot_th_cf(map_type='EE', with_beam=False)
    plt.loglog(l, l*(l+1)*cl_interp_e * bl**2 / (2*np.pi), label='interp EE')
    plot_th_cf(map_type='BB', with_beam=False)
    plt.loglog(l, l*(l+1)*cl_interp_b * bl**2 / (2*np.pi), label='interp BB')
    # plt.loglog(l, l*(l+1)*cl_interp_b*bl**2 / (2*np.pi), label='interp BB with beam')
    plt.xlabel('$\\ell$')
    plt.ylabel('$D_\\ell [\\mu K^2]$')
    plt.title(f'Power spectrum @ {freq}GHz (with beam convolved)')
    plt.legend()
    plt.show()

def gen_interp_cov():
    cl_fg = gen_fg_cl()
    cl_cmb = gen_cmb_cl(beam=beam, lmax=lmax)

    cl_tot = cl_fg + cl_cmb

    lmax_cov = 2200
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax_cov)
    pcfn, cfn, cf, n = gen_map(rlz_idx=rlz_idx)
    cl_interp_e = np.load('./component_dl/cl_interp_pcfn/e.npy')
    cl_interp_b = np.load('./component_dl/cl_interp_pcfn/b.npy')

    flux_idx = 0
    pix_ind = np.load(f'./pix_idx_qu/{flux_idx}.npy')
    obj_cov = CovCalculator(nside=nside, lmin=2, lmax=lmax_cov, Cl_TT=cl_tot[0], Cl_EE=cl_interp_e*bl**2, Cl_BB=cl_interp_b*bl**2, Cl_TE=cl_tot[3], pixind=pix_ind, calc_opt='polarization', out_pol_opt='QU')
    MP = obj_cov.run_calc_cov()

    path_cov = Path('./cmb_qu_cov_interp')
    path_cov.mkdir(exist_ok=True, parents=True)
    np.save(Path(path_cov / f'{flux_idx}.npy'), MP)

def fit_my_ps():
    lmax_cov = 2200
    pcfn, cfn, cf, n = gen_map(rlz_idx=rlz_idx)
    nstd = np.load(f'../../FGSim/NSTDNORTH/2048/{freq}.npy')
    obj = FitPolPS(m_q=pcfn[1], m_u=pcfn[2], freq=freq, nstd_q=nstd[1], nstd_u=nstd[2], flux_idx=0, df_mask=df, df_ps=df, lmax=lmax_cov, nside=nside, radius_factor=1.5, beam=beam, cov_path='./cmb_qu_cov_interp')

    # obj.calc_definite_fixed_cmb_cov()
    # obj.calc_covariance_matrix(mode='cmb+noise')

    num_ps, chi2dof, fit_P, fit_P_err, fit_phi, fit_phi_err = obj.fit_all(cov_mode='cmb+noise')

    path_res = Path('./parameter/estimate_pcfn')
    path_res.mkdir(exist_ok=True, parents=True)
    np.save(path_res / Path(f'fit_P_{rlz_idx}.npy'), fit_P)
    np.save(path_res / Path(f'fit_phi_{rlz_idx}.npy'), fit_phi)

if __name__ == "__main__":
    # gen_ps_fg_map()
    # check_component()


    # plot_th_cf(map_type='BB')
    # plt.legend()
    # plt.show()

    # path_one_rlz = Path(f'./component_dl/one_rlz_map')
    # path_one_rlz.mkdir(exist_ok=True, parents=True)
    # pcfn, cfn, cf, n = gen_map(rlz_idx=rlz_idx)
    # np.save(path_one_rlz / Path('pcfn.npy'), pcfn)
    # np.save(path_one_rlz / Path('cfn.npy'), cfn)
    # np.save(path_one_rlz / Path('cf.npy'), cf)
    # np.save(path_one_rlz / Path('n.npy'), n)

    # calc_th_cov()
    # do_th_fit()

    # interp_dl(lmax_cov=2200, lmax_interp=lmax)
    # check_interp_cl()

    # gen_interp_cov()
    fit_my_ps()

    pass





