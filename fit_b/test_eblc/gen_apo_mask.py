import numpy as np
import healpy as hp
import pymaster as nmt
import matplotlib.pyplot as plt

from pathlib import Path
from eblc_base_slope import EBLeakageCorrection

nside = 512
npix = hp.nside2npix(nside=nside)
beam = 67
mask = np.load('../../src/mask/north/BINMASKG.npy')
# m = np.load('../../fitdata/synthesis_data/2048/CMBNOISE/270/1.npy')
rlz_idx=0
cmb_seed = np.load('../seeds_cmb_2k.npy')
noise_seed = np.load('../seeds_noise_2k.npy')
fg_seed = np.load('../seeds_fg_2k.npy')

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
    scalar_field = nmt.NmtField(apo_mask, [scalar_map], beam=bl, masked_on_input=masked_on_input)
    dl = nmt.compute_full_master(scalar_field, scalar_field, bin_dl)
    return dl[0]

lmax = 500
bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=7000, pol=True)[:,2]
l_min_edges, l_max_edges = generate_bins(l_min_start=30, delta_l_min=30, l_max=lmax, fold=0.2)
bin_dl = nmt.NmtBin.from_edges(l_min_edges, l_max_edges, is_Dell=True)
ell_arr = bin_dl.get_effective_ells()

def gen_fg_cl():
    Cl_TT = np.load('../Cl_fg/data_apo_lmaxout/cl_fg_TT.npy')
    Cl_EE = np.load('../Cl_fg/data_apo_lmaxout/cl_fg_EE.npy')
    Cl_BB = np.load('../Cl_fg/data_apo_lmaxout/cl_fg_BB.npy')
    Cl_TE = np.zeros_like(Cl_TT)
    return np.array([Cl_TT, Cl_EE, Cl_BB, Cl_TE])

def gen_fg_cl_1():
    Cl_TT = np.load('../Cl_fg/data/cl_fg_TT.npy')
    Cl_EE = np.load('../Cl_fg/data/cl_fg_EE.npy')
    Cl_BB = np.load('../Cl_fg/data/cl_fg_BB.npy')
    Cl_TE = np.zeros_like(Cl_TT)
    return np.array([Cl_TT, Cl_EE, Cl_BB, Cl_TE])

def gen_fg_cl_2():
    Cl_TT = np.load('../Cl_fg/data_old/cl_fg_TT.npy')
    Cl_EE = np.load('../Cl_fg/data_old/cl_fg_EE.npy')
    Cl_BB = np.load('../Cl_fg/data_old/cl_fg_BB.npy')
    Cl_TE = np.zeros_like(Cl_TT)
    return np.array([Cl_TT, Cl_EE, Cl_BB, Cl_TE])

def gen_fg_cl_3():
    Cl_TT = np.load('../Cl_fg/data_full_lmax3nside/cl_fg_TT.npy')
    Cl_EE = np.load('../Cl_fg/data_full_lmax3nside/cl_fg_EE.npy')
    Cl_BB = np.load('../Cl_fg/data_full_lmax3nside/cl_fg_BB.npy')
    Cl_TE = np.zeros_like(Cl_TT)
    return np.array([Cl_TT, Cl_EE, Cl_BB, Cl_TE])

def gen_map(lmax, component):

    if component == 'c':
        cls = np.load('../../src/cmbsim/cmbdata/cmbcl_8k.npy').T
        np.random.seed(seed=cmb_seed[rlz_idx])
        cmb_iqu = hp.synfast(cls=cls, nside=nside, fwhm=np.deg2rad(beam)/60, lmax=3*nside-1, new=True)
        return cmb_iqu

    elif component == '10c':
        cls = 10 * np.load('../../src/cmbsim/cmbdata/cmbcl_8k.npy').T
        np.random.seed(seed=cmb_seed[rlz_idx])
        cmb_iqu = hp.synfast(cls=cls, nside=nside, fwhm=np.deg2rad(beam)/60, lmax=3*nside-1, new=True)
        return cmb_iqu

    elif component == '1000c':
        cls = 1000 * np.load('../../src/cmbsim/cmbdata/cmbcl_8k.npy').T
        np.random.seed(seed=cmb_seed[rlz_idx])
        cmb_iqu = hp.synfast(cls=cls, nside=nside, fwhm=np.deg2rad(beam)/60, lmax=3*nside-1, new=True)
        return cmb_iqu

    elif component == 'cn':
        cls = np.load('../../src/cmbsim/cmbdata/cmbcl_8k.npy').T
        np.random.seed(seed=cmb_seed[rlz_idx])
        cmb_iqu = hp.synfast(cls=cls, nside=nside, fwhm=np.deg2rad(beam)/60, lmax=3*nside-1, new=True)

        nstd = np.load('../../FGSim/NSTDNORTH/512/30.npy')
        np.random.seed(seed=noise_seed[rlz_idx])
        noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))
        return cmb_iqu + noise

    elif component == 'n':
        nstd = np.load('../../FGSim/NSTDNORTH/512/30.npy')
        np.random.seed(seed=noise_seed[rlz_idx])
        noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))
        return noise

    elif component == 'f':
        cls_fg = gen_fg_cl()
        np.random.seed(seed=fg_seed[rlz_idx])
        m_fg = hp.synfast(cls=cls_fg, nside=nside, fwhm=0, new=True, lmax=600)
        return m_fg

    elif component == 'cfn':
        cls = np.load('../../src/cmbsim/cmbdata/cmbcl_8k.npy').T
        np.random.seed(seed=cmb_seed[rlz_idx])
        cmb_iqu = hp.synfast(cls=cls, nside=nside, fwhm=np.deg2rad(beam)/60, lmax=3*nside-1, new=True)

        nstd = np.load('../../FGSim/NSTDNORTH/512/30.npy')
        np.random.seed(seed=noise_seed[rlz_idx])
        noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))

        cls_fg = gen_fg_cl()
        np.random.seed(seed=fg_seed[rlz_idx])
        m_fg = hp.synfast(cls=cls_fg, nside=nside, fwhm=0, new=True, lmax=600)
        return m_fg + noise + cmb_iqu

def test_fg_dl():
    fg_cls = gen_fg_cl()
    fg_cls_1 = gen_fg_cl_1()
    fg_cls_2 = gen_fg_cl_2()
    fg_cls_3 = gen_fg_cl_3()
    m_n = gen_map(lmax=1000, component='n')
    m_c = gen_map(lmax=1000, component='c')
    n_cls = hp.anafast(m_n, lmax=600)
    c_cls = hp.anafast(m_c, lmax=600)
    l = np.arange(601)
    # plt.loglog(l*(l+1)*fg_cls[2]/bl[:601]**2/(2*np.pi), label='lmax out')
    plt.loglog(l*(l+1)*fg_cls_1[2]/bl[:601]**2/(2*np.pi), label='full sky fg(not gaussian)')
    # plt.loglog(l*(l+1)*fg_cls_2[2]/bl[:601]**2/(2*np.pi), label='previous data diffuse fg')
    plt.loglog(l*(l+1)*fg_cls_3[2]/bl[:601]**2/(2*np.pi), label='from B map')
    plt.loglog(l*(l+1)*fg_cls_3[2]/(2*np.pi), label='from B map no debeam')
    plt.loglog(l*(l+1)*n_cls[2]/bl[:601]**2/(2*np.pi), label='noise')
    plt.loglog(l*(l+1)*c_cls[2]/bl[:601]**2/(2*np.pi), label='cmb')
    # plt.loglog(l*(l+1)*(c_cls[2]+n_cls[2]+fg_cls_2[2])/bl[:601]**2/(2*np.pi), label='cfn')
    plt.legend()
    plt.xlabel('$\\ell$')
    plt.ylabel('$D_\\ell$')
    plt.show()

# test_fg_dl()

def test_cmb_dl():
    lmax = 1500
    l = np.arange(lmax+1)
    apo_mask = np.load('./mask/apo_mask_2.npy')
    fsky = np.sum(apo_mask) / np.size(apo_mask)
    m_c = gen_map(lmax=1000, component='c')
    c_cls = hp.anafast(m_c, lmax=lmax)
    m_c_b = hp.alm2map(hp.map2alm(m_c)[2], nside=nside)

    c_cl = hp.anafast(m_c_b * apo_mask)[0:lmax+1] / fsky
    cls = np.load('../../src/cmbsim/cmbdata/cmbcl_8k.npy').T
    plt.loglog(l*(l+1)*c_cl/bl[:lmax+1]**2/(2*np.pi), label='cmb partial')
    plt.loglog(l*(l+1)*c_cls[2]/bl[:lmax+1]**2/(2*np.pi), label='cmb full')
    plt.loglog(l*(l+1)*c_cl/(2*np.pi), label='cmb partial no debeam')
    plt.loglog(l*(l+1)*c_cls[2]/(2*np.pi), label='cmb full no debeam')
    plt.loglog(l*(l+1)*cls[2,:lmax+1]/(2*np.pi), label='cmb true')
    # plt.loglog(bl[:lmax+1]**2, label='bl square')
    plt.legend()
    plt.xlabel('$\\ell$')
    plt.ylabel('$D_\\ell$')
    plt.ylim(1e-12,1e1)
    plt.show()

# test_cmb_dl()

def gen_apo_mask():
    apo_mask = nmt.mask_apodization(mask_in=mask, aposize=5, apotype='C1')
    path_mask = Path('./mask')
    path_mask.mkdir(exist_ok=True, parents=True)
    # np.save(path_mask / Path('bin_mask.npy'), mask)
    np.save(path_mask / Path('apo_mask_5.npy'), apo_mask)

# gen_apo_mask()

def check_mask():
    apo_mask = np.load('./mask/apo_mask.npy')
    hp.orthview(mask, rot=[100,50,0], half_sky=True)
    hp.orthview(apo_mask, rot=[100,50,0], half_sky=True)
    plt.show()

# check_mask()

def save_map():
    c = gen_map(lmax=1000, component='c')
    f = gen_map(lmax=1000, component='f')
    n = gen_map(lmax=1000, component='n')
    cfn = gen_map(lmax=1000, component='cfn')

    path_map = Path('./maps')
    path_map.mkdir(exist_ok=True, parents=True)
    np.save(path_map / Path(f'c_{rlz_idx}.npy'), c)
    np.save(path_map / Path(f'f_{rlz_idx}.npy'), f)
    np.save(path_map / Path(f'n_{rlz_idx}.npy'), n)
    np.save(path_map / Path(f'cfn_{rlz_idx}.npy'), cfn)

# save_map()

def check_eblc():
    apo_mask = np.load('./mask/apo_mask_2.npy')
    m = gen_map(lmax=1000, component='1000c')
    m_b = hp.alm2map(hp.map2alm(m)[2], nside=nside)

    obj = EBLeakageCorrection(m=m, lmax=3*nside-1, nside=nside, mask=mask, post_mask=mask)
    _,_,cln_b = obj.run_eblc()

    dl_cln = calc_dl_from_scalar_map(cln_b, bl, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=False)
    dl_true = calc_dl_from_scalar_map(m_b, bl, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=False)
    path_dl = Path('./dls/1000c_2/cutqufitqu')
    path_dl.mkdir(exist_ok=True, parents=True)
    np.save(path_dl / Path(f'{rlz_idx}.npy'), dl_cln)
    np.save(path_dl / Path(f'../{rlz_idx}.npy'), dl_true)

# check_eblc()

def check_dl():
    true_list = []
    dl_list_1 = []
    nsim = 500
    for rlz_idx in range(500):
        dl_true = np.load(f'./dls/1000c_2/{rlz_idx}.npy')
        dl_1 = np.load(f'./dls/1000c_2/cutqufitqu/{rlz_idx}.npy')
        true_list.append(dl_true)
        dl_list_1.append(dl_1)

    true_arr = np.array(true_list)
    dl_arr_1 = np.array(dl_list_1)
    true_mean = np.mean(true_arr, axis=0)
    dl_mean_1 = np.mean(dl_arr_1, axis=0)

    # rmse_1 = np.sqrt(np.sum((dl_arr_1-true_arr) ** 2, axis=0) / nsim)
    plt.figure(1)
    plt.scatter(ell_arr, true_mean, label='true', marker='.')
    plt.scatter(ell_arr, dl_mean_1, label='cutqufitqu', marker='.')

    plt.xlabel('$\\ell$')
    plt.ylabel('$D_\\ell^{BB}$')
    plt.loglog()
    plt.legend()
    plt.title('mean over 500 simulations')

    plt.show()
    plt.figure(2)
    plt.scatter(ell_arr, np.abs(dl_mean_1-true_mean)/true_mean)
    plt.loglog()
    plt.xlabel('$\\ell$')
    plt.ylabel('relative error')
    plt.title('bias')

    plt.show()

# check_dl()

def check_fg_bias():
    apo_mask = np.load('./mask/apo_mask_2.npy')
    m_cfn = gen_map(lmax=1000, component='cfn')
    m_cfn_b = hp.alm2map(hp.map2alm(m_cfn)[2], nside=nside)

    m_c = gen_map(lmax=1000, component='c')
    m_c_b = hp.alm2map(hp.map2alm(m_c)[2], nside=nside)

    obj = EBLeakageCorrection(m=m_cfn, lmax=3*nside-1, nside=nside, mask=mask, post_mask=mask)
    _,_,cln_cfn_b = obj.run_eblc()
    slope_cfn = obj.return_slope()

    dl_cln_cfn = calc_dl_from_scalar_map(cln_cfn_b, bl, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=False)
    dl_cfn_true = calc_dl_from_scalar_map(m_cfn_b, bl, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=False)

    obj_c = EBLeakageCorrection(m=m_c, lmax=3*nside-1, nside=nside, mask=mask, post_mask=mask)
    _,_,cln_c_b = obj_c.run_eblc()
    slope_c = obj_c.return_slope()

    obj_c_1 = EBLeakageCorrection(m=m_c, lmax=3*nside-1, nside=nside, mask=mask, post_mask=mask, slope_in=slope_cfn)
    _,_,cln_c_b_1 = obj_c_1.run_eblc()

    dl_cln_c = calc_dl_from_scalar_map(cln_c_b, bl, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=False)
    dl_cln_c_1 = calc_dl_from_scalar_map(cln_c_b_1, bl, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=False)
    dl_c_true = calc_dl_from_scalar_map(m_c_b, bl, apo_mask=apo_mask, bin_dl=bin_dl, masked_on_input=False)

    path_dl = Path('./dls/cfn/cutqufitqu')
    path_dl.mkdir(exist_ok=True, parents=True)
    np.save(path_dl / Path(f'cln_cfn_{rlz_idx}.npy'), dl_cln_cfn)
    np.save(path_dl / Path(f'cln_c_{rlz_idx}.npy'), dl_cln_c)
    np.save(path_dl / Path(f'cln_c_slope_{rlz_idx}.npy'), dl_cln_c_1)
    np.save(path_dl / Path(f'../cfn_{rlz_idx}.npy'), dl_cfn_true)
    np.save(path_dl / Path(f'../c_{rlz_idx}.npy'), dl_c_true)
    np.save(path_dl / Path(f'slope_cfn_{rlz_idx}.npy'), slope_cfn)
    np.save(path_dl / Path(f'slope_c_{rlz_idx}.npy'), slope_c)

# check_fg_bias()

def check_bias_dl():
    cfn_true_list = []
    c_true_list = []
    cln_cfn_list = []
    cln_c_list = []
    cln_c_slope_list = []
    nsim = 500
    for rlz_idx in range(500):
        dl_cfn_true = np.load(f'./dls/cfn/cfn_{rlz_idx}.npy')
        dl_c_true = np.load(f'./dls/cfn/c_{rlz_idx}.npy')
        dl_cln_cfn = np.load(f'./dls/cfn/cutqufitqu/cln_cfn_{rlz_idx}.npy')
        dl_cln_c = np.load(f'./dls/cfn/cutqufitqu/cln_c_{rlz_idx}.npy')
        dl_cln_c_slope = np.load(f'./dls/cfn/cutqufitqu/cln_c_slope_{rlz_idx}.npy')
        cfn_true_list.append(dl_cfn_true)
        c_true_list.append(dl_c_true)
        cln_cfn_list.append(dl_cln_cfn)
        cln_c_list.append(dl_cln_c)
        cln_c_slope_list.append(dl_cln_c_slope)

    cfn_true_arr = np.array(cfn_true_list)
    c_true_arr = np.array(c_true_list)
    cln_cfn_arr = np.array(cln_cfn_list)
    cln_c_arr = np.array(cln_c_list)
    cln_c_slope_arr = np.array(cln_c_slope_list)

    cfn_true_mean = np.mean(cfn_true_arr, axis=0)
    c_true_mean = np.mean(c_true_arr, axis=0)
    cln_cfn_mean = np.mean(cln_cfn_arr, axis=0)
    cln_c_mean = np.mean(cln_c_arr, axis=0)
    cln_c_slope_mean = np.mean(cln_c_slope_arr, axis=0)


    # rmse_1 = np.sqrt(np.sum((dl_arr_1-true_arr) ** 2, axis=0) / nsim)
    plt.figure(1)
    plt.scatter(ell_arr, cfn_true_mean, label='cfn true', marker='.')
    plt.scatter(ell_arr, c_true_mean, label='c true', marker='.')
    plt.scatter(ell_arr, cln_cfn_mean, label='cln cfn', marker='.')
    plt.scatter(ell_arr, cln_c_mean, label='cln c', marker='.')
    plt.scatter(ell_arr, cln_c_slope_mean, label='cln c slope', marker='.')

    plt.xlabel('$\\ell$')
    plt.ylabel('$D_\\ell^{BB}$')
    plt.loglog()
    plt.legend()
    plt.title('mean over 500 simulations')
    plt.show()

    plt.figure(2)
    plt.scatter(ell_arr, np.abs(cln_c_mean-c_true_mean)/c_true_mean, label='cmb slope')
    plt.scatter(ell_arr, np.abs(cln_c_slope_mean-c_true_mean)/c_true_mean, label='cfn slope')
    plt.legend()
    plt.loglog()
    plt.xlabel('$\\ell$')
    plt.ylabel('relative error')
    plt.title('bias')
    plt.show()

    plt.figure(3)
    plt.scatter(ell_arr, np.abs(cln_cfn_mean-cfn_true_mean)/cfn_true_mean, label='relative error')
    plt.legend()
    plt.loglog()
    plt.xlabel('$\\ell$')
    plt.ylabel('relative error')
    plt.title('bias')

    plt.show()

check_bias_dl()

def check_factor():
    for i in range(200):
        cfn_factor = np.load(f'./dls/cfn/cutqufitqu/slope_cfn_{i}.npy')
        c_factor = np.load(f'./dls/cfn/cutqufitqu/slope_c_{i}.npy')
        print(f'{cfn_factor=}, {c_factor=}')

# check_factor()


