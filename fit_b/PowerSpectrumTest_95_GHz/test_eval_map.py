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
ori_mask = np.load('../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5.npy')
ori_apo_mask = np.load('../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5APO_5.npy')
n_ps_list = np.linspace(10, 135, 8, dtype=int)
print(f'{n_ps_list=}')

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

def gen_fg_cl():
    cl_fg = np.load('./data/debeam_full_b/cl_fg.npy')
    Cl_TT = cl_fg[0]
    Cl_EE = cl_fg[1]
    Cl_BB = cl_fg[2]
    Cl_TE = np.zeros_like(Cl_TT)
    return np.array([Cl_TT, Cl_EE, Cl_BB, Cl_TE])

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

# Part 2: see theoretical power spectrum
def plot_th_cf(map_type):
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
        # plt.loglog(l*(l+1)*cl_th_fg[i]/bl[i]**2/(2*np.pi), label=f'fg, {i=}')
        # plt.loglog(l*(l+1)*cl_th_cmb[i]/bl[i]**2/(2*np.pi), label=f'cmb, {i=}')
        plt.loglog(l*(l+1)*(cl_th_cmb[i]+cl_th_fg[i])/bl[i]**2/(2*np.pi), label=f'cmb+fg input EE')
        # plt.legend()
        # plt.show()
    elif map_type == "BB":
        i = 2
        plt.loglog(l*(l+1)*(cl_th_cmb[i]+cl_th_fg[i])/bl[i]**2/(2*np.pi), label=f'cmb+fg input BB')

# Part 3: see how power spectrum change with diffrent band power

# delta_l_min_list = [4, 6, 8, 10, 15]
# fold_list = [0.03, 0.05, 0.08, 0.1, 0.2]

def estimate_cl():
    m_cfn = np.load('./cfn.npy')
    m_q_cfn = m_cfn[1].copy()
    m_u_cfn = m_cfn[2].copy()
    m_n = np.load('./n.npy')
    m_q_n = m_n[1].copy()
    m_u_n = m_n[2].copy()

    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax, pol=True)[:,2]

    path_res = Path('./result')
    path_res.mkdir(exist_ok=True, parents=True)

    for delta_l_min in delta_l_min_list:
        for fold in fold_list:
            print(f'./{delta_l_min=}, {fold=}')
            l_min_edges, l_max_edges = generate_bins(l_min_start=20, delta_l_min=delta_l_min, l_max=lmax+1, fold=fold)
            bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
            ell_arr = bin_dl.get_effective_ells()
            dl_cfn = calc_dl_from_pol_map(m_q=m_q_cfn, m_u=m_u_cfn, bl=bl, apo_mask=ori_apo_mask, bin_dl=bin_dl, masked_on_input=False, purify_b=True)
            dl_n = calc_dl_from_pol_map(m_q=m_q_n, m_u=m_u_n, bl=bl, apo_mask=ori_apo_mask, bin_dl=bin_dl, masked_on_input=False, purify_b=True)
            np.save(path_res / Path(f"ell_arr_{delta_l_min}_{fold}.npy"), ell_arr)
            np.save(path_res / Path(f"cfn_{delta_l_min}_{fold}.npy"), dl_cfn)
            np.save(path_res / Path(f"n_{delta_l_min}_{fold}.npy"), dl_n)

def check_estimated_cl():
    # plot_th_cf(map_type='BB')
    # for delta_l_min in delta_l_min_list:
    delta_l_min=10
    for delta_l_min in delta_l_min_list:
        plot_th_cf(map_type='BB')
        for fold in fold_list:
            ell_arr = np.load(f'./result/ell_arr_{delta_l_min}_{fold}.npy')
            dl_cfn = np.load(f'./result/cfn_{delta_l_min}_{fold}.npy')[3]
            dl_n = np.load(f'./result/n_{delta_l_min}_{fold}.npy')[3]
            plt.loglog(ell_arr, dl_cfn - dl_n, label=f'delta_l={delta_l_min}, {fold=}')

            plt.xlabel('$\\ell$')
            plt.ylabel('$D_\\ell^{BB} [\\mu K^2]$')
        plt.legend()
        plt.show()

# Part 4: See the distribution of point sources flux density
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


def dl2cl(D_ell):

    ell = np.arange(len(D_ell))
    mask = ell > 1
    C_ell = np.zeros_like(D_ell, dtype=np.float64)
    C_ell[mask] = (2 * np.pi * D_ell[mask]) / (ell[mask] * (ell[mask] + 1))
    C_ell[~mask] = 0
    return C_ell


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

def interp_dl_example():
    ell_arr = np.load('./result/ell_arr_10_0.1.npy')
    band_power = np.load('./result/cfn_10_0.1.npy')[0] - np.load('./result/n_10_0.1.npy')[0]
    lmax_cov = 600

    l, dl_interp = interp_band_power(ell_arr=ell_arr, band_power=band_power, lmax_cov=lmax_cov)
    # plot_th_cf(map_type='EE')
    cl_interp = dl2cl(dl_interp)
    plt.loglog(l, dl_interp, label='dl interpolation')
    plt.loglog(l, cl_interp, label='cl interpolation')
    plt.show()

def interp_dl():

    pcfn, cfn, cf, n = gen_map(rlz_idx=rlz_idx)

    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax, pol=True)[:,2]
    l_min_edges, l_max_edges = generate_bins(l_min_start=20, delta_l_min=10, l_max=lmax+1, fold=0.1)
    bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    ell_arr = bin_dl.get_effective_ells()

    dl_cfn = calc_dl_from_pol_map(m_q=pcfn[1].copy(), m_u=pcfn[2].copy(), bl=bl, apo_mask=ori_apo_mask, bin_dl=bin_dl, masked_on_input=False, purify_b=True)
    dl_n = calc_dl_from_pol_map(m_q=n[1].copy(), m_u=n[2].copy(), bl=bl, apo_mask=ori_apo_mask, bin_dl=bin_dl, masked_on_input=False, purify_b=True)


    lmax_cov = 600

    l, dl_interp_e = interp_band_power(ell_arr=ell_arr, band_power=dl_cfn[0]-dl_n[0], lmax_cov=lmax_cov)
    l, dl_interp_b = interp_band_power(ell_arr=ell_arr, band_power=dl_cfn[3]-dl_n[3], lmax_cov=lmax_cov)
    # plot_th_cf(map_type='EE')
    cl_interp_e = dl2cl(dl_interp_e)
    cl_interp_b = dl2cl(dl_interp_b)

    path_cl_interp = Path('./cl_interp_pcfn')
    path_cl_interp.mkdir(exist_ok=True, parents=True)
    np.save(path_cl_interp / Path(f'e.npy'), cl_interp_e)
    np.save(path_cl_interp / Path(f'b.npy'), cl_interp_b)


    # plt.loglog(l, dl_interp, label='dl interpolation')
    # plt.loglog(l, cl_interp, label='cl interpolation')
    # plt.show()

def check_interp_dl():
    cl_interp_e = np.load('./cl_interp/e.npy')
    cl_interp_b = np.load('./cl_interp/b.npy')
    l = np.arange(len(cl_interp_b))
    plot_th_cf(map_type='EE')
    plt.loglog(l*(l+1)*cl_interp_e/(2*np.pi), label='cl interpolate EE')
    plot_th_cf(map_type='BB')
    plt.loglog(l*(l+1)*cl_interp_b/(2*np.pi), label='cl interpolate BB')
    plt.legend()
    plt.xlabel('$\\ell$')
    plt.ylabel('$D_\\ell [\\mu K^2]$')
    plt.show()


def fit_my_ps():
    cl_fg = gen_fg_cl()
    cl_cmb = gen_cmb_cl(beam=beam, lmax=lmax)

    cl_tot = cl_fg + cl_cmb

    lmax_cov = 600
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax_cov)
    pcfn, cfn, cf, n = gen_map(rlz_idx=rlz_idx)
    cl_interp_e = np.load('./cl_interp_pcfn/e.npy')
    cl_interp_b = np.load('./cl_interp_pcfn/b.npy')

    flux_idx = 0
    pix_ind = np.load(f'./pix_idx_qu/{flux_idx}.npy')
    obj_cov = CovCalculator(nside=nside, lmin=2, lmax=lmax_cov, Cl_TT=cl_tot[0], Cl_EE=cl_interp_e*bl**2, Cl_BB=cl_interp_b*bl**2, Cl_TE=cl_tot[3], pixind=pix_ind, calc_opt='polarization', out_pol_opt='QU')
    MP = obj_cov.run_calc_cov()

    path_cov = Path('./cmb_qu_cov')
    path_cov.mkdir(exist_ok=True, parents=True)
    # np.save(f'./test_class_cov/cmb.npy', MP)
    np.save(Path(path_cov / f'{flux_idx}.npy'), MP)

    nstd = np.load(f'../../FGSim/NSTDNORTH/2048/{freq}.npy')
    obj = FitPolPS(m_q=pcfn[1], m_u=pcfn[2], freq=freq, nstd_q=nstd[1], nstd_u=nstd[2], flux_idx=0, df_mask=df, df_ps=df, lmax=lmax_cov, nside=nside, radius_factor=1.5, beam=beam)

    # obj.calc_definite_fixed_cmb_cov()
    # obj.calc_covariance_matrix(mode='cmb+noise')

    num_ps, chi2dof, fit_P, fit_P_err, fit_phi, fit_phi_err = obj.fit_all(cov_mode='cmb+noise')

    path_res = Path('./parameter/estimate_pcfn')
    path_res.mkdir(exist_ok=True, parents=True)
    np.save(path_res / Path(f'fit_P_{rlz_idx}.npy'), fit_P)
    np.save(path_res / Path(f'fit_phi_{rlz_idx}.npy'), fit_phi)

def interp_ps_dl():

    pcfn, cfn, cf, n = gen_map(rlz_idx=rlz_idx)
    mask = np.load('./ps_mask/81_ps_mask.npy')

    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax, pol=True)[:,2]
    l_min_edges, l_max_edges = generate_bins(l_min_start=20, delta_l_min=10, l_max=lmax+1, fold=0.1)
    bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
    ell_arr = bin_dl.get_effective_ells()

    dl_cfn = calc_dl_from_pol_map(m_q=pcfn[1].copy(), m_u=pcfn[2].copy(), bl=bl, apo_mask=mask, bin_dl=bin_dl, masked_on_input=False, purify_b=True)
    dl_n = calc_dl_from_pol_map(m_q=n[1].copy(), m_u=n[2].copy(), bl=bl, apo_mask=mask, bin_dl=bin_dl, masked_on_input=False, purify_b=True)

    lmax_cov = 600

    l, dl_interp_e = interp_band_power(ell_arr=ell_arr, band_power=dl_cfn[0]-dl_n[0], lmax_cov=lmax_cov)
    l, dl_interp_b = interp_band_power(ell_arr=ell_arr, band_power=dl_cfn[3]-dl_n[3], lmax_cov=lmax_cov)
    cl_interp_e = dl2cl(dl_interp_e)
    cl_interp_b = dl2cl(dl_interp_b)

    path_cl_interp = Path('./cl_interp_ps')
    path_cl_interp.mkdir(exist_ok=True, parents=True)
    np.save(path_cl_interp / Path(f'e.npy'), cl_interp_e)
    np.save(path_cl_interp / Path(f'b.npy'), cl_interp_b)


    plot_th_cf(map_type='EE')
    plt.loglog(l, dl_interp_e, label='dl interpolation, EE')
    plot_th_cf(map_type='BB')
    plt.loglog(l, dl_interp_b, label='dl interpolation, BB')
    plt.legend()

    plt.xlabel('$\\ell$')
    plt.ylabel('$D_\\ell [\\mu K^2]$')
    plt.show()



if __name__ == "__main__":
    # estimate_cl()
    # check_estimated_cl()
    # interp_dl_example()
    # interp_dl()
    # check_interp_dl()
    fit_my_ps()
    # interp_ps_dl()





