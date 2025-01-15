import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pandas as pd
import time
import pickle
import os,sys
import logging

from pathlib import Path
from iminuit import Minuit
from iminuit.cost import LeastSquares
from fit_qu_no_const import FitPolPS
from config import lmax, nside, freq, beam, ps_number
from eblc_base_slope import EBLeakageCorrection

# logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s -%(name)s - %(message)s')
logging.basicConfig(level=logging.WARNING)
# logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.INFO)

def gen_fg_cl():
    cl_fg = np.load('./data/debeam_full_b/cl_fg.npy')
    Cl_TT = cl_fg[0]
    Cl_EE = cl_fg[1]
    Cl_BB = cl_fg[2]
    Cl_TE = np.zeros_like(Cl_TT)
    return np.array([Cl_TT, Cl_EE, Cl_BB, Cl_TE])

def gen_map(beam, freq, lmax, rlz_idx=0, mode='mean', return_noise=False):
    # mode can be mean or std
    noise_seed = np.load('../seeds_noise_2k.npy')
    cmb_seed = np.load('../seeds_cmb_2k.npy')
    nside = 2048

    nstd = np.load(f'../../FGSim/NSTDNORTH/2048/{freq}.npy')
    npix = hp.nside2npix(nside=2048)
    np.random.seed(seed=noise_seed[rlz_idx])
    # noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))
    noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))
    print(f"{np.std(noise[1])=}")

    if return_noise:
        return noise

    ps = np.load(f'../../fitdata/2048/PS/{freq}/ps.npy')
    fg = np.load(f'../../fitdata/2048/FG/{freq}/fg.npy')

    cls = np.load('../../src/cmbsim/cmbdata/cmbcl_8k.npy')
    if mode=='std':
        np.random.seed(seed=cmb_seed[rlz_idx])
    elif mode=='mean':
        np.random.seed(seed=cmb_seed[0])

    cmb_iqu = hp.synfast(cls.T, nside=nside, fwhm=np.deg2rad(beam)/60, new=True, lmax=3*nside-1)

    m = noise + ps + cmb_iqu + fg
    return m

def gen_map_all(beam, freq, lmax, rlz_idx=0, mode='mean', return_noise=False):
    # mode can be mean or std
    noise_seed = np.load('../seeds_noise_2k.npy')
    cmb_seed = np.load('../seeds_cmb_2k.npy')
    nside = 2048

    nstd = np.load(f'../../FGSim/NSTDNORTH/2048/{freq}.npy')
    npix = hp.nside2npix(nside=2048)
    np.random.seed(seed=noise_seed[rlz_idx])
    # noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))
    noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))
    print(f"{np.std(noise[1])=}")

    if return_noise:
        return noise

    ps = np.load(f'../../fitdata/2048/PS/{freq}/ps.npy')
    fg = np.load(f'../../fitdata/2048/FG/{freq}/fg.npy')

    cls = np.load('../../src/cmbsim/cmbdata/cmbcl_8k.npy')
    if mode=='std':
        np.random.seed(seed=cmb_seed[rlz_idx])
    elif mode=='mean':
        np.random.seed(seed=cmb_seed[0])

    cmb_iqu = hp.synfast(cls.T, nside=nside, fwhm=np.deg2rad(beam)/60, new=True, lmax=3*nside-1)

    pcfn = noise + ps + cmb_iqu + fg
    cfn = noise + cmb_iqu + fg
    cf = cmb_iqu + fg
    return pcfn, cfn, cf, noise



def gen_pix_idx(flux_idx=0):
    npix = hp.nside2npix(nside)

    time0 = time.perf_counter()
    nstd = np.load(f'../../FGSim/NSTDNORTH/2048/{freq}.npy')
    print(f'{nstd[1,0]=}')
    nstd_q = nstd[1].copy()
    nstd_u = nstd[2].copy()
    m_q = nstd_q * np.random.normal(loc=0, scale=1, size=(npix,))
    m_u = nstd_u * np.random.normal(loc=0, scale=1, size=(npix,))

    logger.info(f'time for fitting = {time.perf_counter()-time0}')

    df_mask = pd.read_csv(f'./mask/{freq}.csv')
    df_ps = df_mask

    # obj = FitPolPS(m_q=m_q, m_u=m_u, freq=freq, nstd_q=nstd_q, nstd_u=nstd_u, flux_idx=flux_idx, df_mask=df_mask, df_ps=df_mask, lmax=lmax, nside=nside, radius_factor=1.5, beam=beam, epsilon=0.00001)
    n_ps = ps_number

    for flux_idx in range(n_ps):
        print(f'{flux_idx=}')
        obj = FitPolPS(m_q=m_q, m_u=m_u, freq=freq, nstd_q=nstd_q, nstd_u=nstd_u, flux_idx=flux_idx, df_mask=df_mask, df_ps=df_mask, lmax=lmax, nside=nside, radius_factor=1.5, beam=beam, epsilon=0.00001)

def gen_cov_inv():
    npix = hp.nside2npix(nside)

    time0 = time.perf_counter()
    nstd = np.load(f'../../FGSim/NSTDNORTH/2048/{freq}.npy')
    print(f'{nstd[1,0]=}')
    nstd_q = nstd[1].copy()
    nstd_u = nstd[2].copy()
    m_q = nstd_q * np.random.normal(loc=0, scale=1, size=(npix,))
    m_u = nstd_u * np.random.normal(loc=0, scale=1, size=(npix,))

    logger.info(f'time for fitting = {time.perf_counter()-time0}')

    df_mask = pd.read_csv(f'./mask/{freq}.csv')
    df_ps = df_mask

    # obj = FitPolPS(m_q=m_q, m_u=m_u, freq=freq, nstd_q=nstd_q, nstd_u=nstd_u, flux_idx=flux_idx, df_mask=df_mask, df_ps=df_mask, lmax=lmax, nside=nside, radius_factor=1.5, beam=beam, epsilon=0.00001)

    flux_idx = 0
    print(f'{flux_idx=}')
    obj = FitPolPS(m_q=m_q, m_u=m_u, freq=freq, nstd_q=nstd_q, nstd_u=nstd_u, flux_idx=flux_idx, df_mask=df_mask, df_ps=df_mask, lmax=lmax, nside=nside, radius_factor=1.5, beam=beam, epsilon=0.00001, cov_path='./cmb_qu_cov_interp')
    obj.calc_definite_fixed_cmb_cov()
    obj.calc_covariance_matrix()

def check_parameter_distribution():
    rlz_idx = 0
    npix = hp.nside2npix(nside)

    time0 = time.perf_counter()
    nstd = np.load(f'../../FGSim/NSTDNORTH/2048/{freq}.npy')
    print(f'{nstd[1,0]=}')
    nstd_q = nstd[1].copy()
    nstd_u = nstd[2].copy()
    # ps = np.load('./data/ps/ps.npy')
    # noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))
    m = gen_map(beam=beam, freq=freq, lmax=lmax, rlz_idx=rlz_idx, mode='std')
    # m = gen_map(beam=beam, freq=freq, lmax=lmax, mode='std')
    m_q = m[1].copy()
    m_u = m[2].copy()
    logger.debug(f'{sys.getrefcount(m_q)-1=}')

    logger.info(f'time for fitting = {time.perf_counter()-time0}')

    df_mask = pd.read_csv(f'./mask/{freq}.csv')
    df_ps = df_mask

    flux_idx=0

    logger.debug(f'{sys.getrefcount(m_q)-1=}')
    obj = FitPolPS(m_q=m_q, m_u=m_u, freq=freq, nstd_q=nstd_q, nstd_u=nstd_u, flux_idx=flux_idx, df_mask=df_mask, df_ps=df_mask, lmax=lmax, nside=nside, radius_factor=1.5, beam=beam, epsilon=0.00001)

    # obj.calc_definite_fixed_cmb_cov()
    # obj.calc_covariance_matrix(mode='cmb+noise')

    # obj.fit_all(cov_mode='cmb+noise')
    num_ps, chi2dof, fit_P, fit_P_err, fit_phi, fit_phi_err = obj.fit_all(cov_mode='cmb+noise')

    # path_res = Path('./parameter/only_noise_vary')
    path_res = Path('./parameter/cmb_noise_vary')
    path_res.mkdir(exist_ok=True, parents=True)
    np.save(path_res / Path(f'fit_P_{rlz_idx}.npy'), fit_P)
    np.save(path_res / Path(f'fit_phi_{rlz_idx}.npy'), fit_phi)

def first_fit_all():
    import gc

    rlz_idx = 0
    npix = hp.nside2npix(nside)

    time0 = time.perf_counter()
    nstd = np.load(f'../../FGSim/NSTDNORTH/2048/{freq}.npy')
    print(f'{nstd[1,0]=}')
    nstd_q = nstd[1].copy()
    nstd_u = nstd[2].copy()
    # ps = np.load('./data/ps/ps.npy')
    # noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))
    m = gen_map(beam=beam, freq=freq, lmax=lmax, rlz_idx=rlz_idx, mode='mean')
    # m = gen_map(beam=beam, freq=freq, lmax=lmax, mode='std')
    m_q = m[1].copy()
    m_u = m[2].copy()
    logger.debug(f'{sys.getrefcount(m_q)-1=}')

    logger.info(f'time for fitting = {time.perf_counter()-time0}')

    df_mask = pd.read_csv(f'./mask/{freq}.csv')
    df_ps = pd.read_csv(f'../../pp_P/mask/ps_csv/{freq}.csv')

    n_ps = ps_number
    logger.debug(f'{sys.getrefcount(m_q)-1=}')
    save_path = Path(f'fit_res/params/mean_for_sigma')
    save_path.mkdir(exist_ok=True, parents=True)

    flux_idx_arr = df_mask.loc[:n_ps-1, 'flux_idx'].to_numpy()
    index_arr = df_mask.loc[:n_ps-1, 'index'].to_numpy()
    lon_arr = df_mask.loc[:n_ps-1, 'lon'].to_numpy()
    lat_arr = df_mask.loc[:n_ps-1, 'lat'].to_numpy()
    iflux_arr = df_mask.loc[:n_ps-1, 'iflux'].to_numpy()
    qflux_arr = df_mask.loc[:n_ps-1, 'qflux'].to_numpy()
    uflux_arr = df_mask.loc[:n_ps-1, 'uflux'].to_numpy()
    pflux_arr = df_mask.loc[:n_ps-1, 'pflux'].to_numpy()

    num_ps_arr = np.zeros_like(flux_idx_arr, dtype=float)
    chi2dof_arr = np.zeros_like(flux_idx_arr, dtype=float)
    true_q_arr = np.zeros_like(flux_idx_arr, dtype=float)
    fit_q_arr = np.zeros_like(flux_idx_arr, dtype=float)
    fit_q_err_arr = np.zeros_like(flux_idx_arr, dtype=float)
    true_u_arr = np.zeros_like(flux_idx_arr, dtype=float)
    fit_u_arr = np.zeros_like(flux_idx_arr, dtype=float)
    fit_u_err_arr = np.zeros_like(flux_idx_arr, dtype=float)


    for flux_idx in range(n_ps):
        logger.debug(f'{flux_idx=}')
        obj = FitPolPS(m_q=m_q, m_u=m_u, freq=freq, nstd_q=nstd_q, nstd_u=nstd_u, flux_idx=flux_idx, df_mask=df_mask, df_ps=df_ps, lmax=lmax, nside=nside, radius_factor=1.5, beam=beam, epsilon=0.00001, threshold_extra_factor=1.5)
        # num_ps, chi2dof, fit_P, fit_P_err, fit_phi, fit_phi_err = obj.fit_all(cov_mode='cmb+noise')
        num_ps, chi2dof, true_q, fit_q, fit_q_err, true_u, fit_u, fit_u_err = obj.fit_all(cov_mode='cmb+noise', return_qu=True)

        del obj
        gc.collect()

        num_ps_arr[flux_idx] = num_ps
        chi2dof_arr[flux_idx] = chi2dof
        true_q_arr[flux_idx] = true_q
        fit_q_arr[flux_idx] = fit_q
        fit_q_err_arr[flux_idx] = fit_q_err
        true_u_arr[flux_idx] = true_u
        fit_u_arr[flux_idx] = fit_u
        fit_u_err_arr[flux_idx] = fit_u_err
        print(f'{fit_q_arr=}')

    df_fit = pd.DataFrame({
        'flux_idx': flux_idx_arr,
        'index': index_arr,
        'lon': lon_arr,
        'lat': lat_arr,
        'iflux': iflux_arr,
        'qflux': qflux_arr,
        'uflux': uflux_arr,
        'pflux': pflux_arr,
        'num_ps': num_ps_arr,
        'chi2dof': chi2dof_arr,
        'true_q': true_q_arr,
        'fit_q': fit_q_arr,
        'fit_q_err': fit_q_err_arr,
        'true_u': true_u_arr,
        'fit_u': fit_u_arr,
        'fit_u_err': fit_u_err_arr
        })

    df_fit.to_csv(f'./mask/{freq}_fit.csv', index=False)


def one_ps_fit():
    rlz_idx = 0
    npix = hp.nside2npix(nside)

    time0 = time.perf_counter()
    nstd = np.load(f'../../FGSim/NSTDNORTH/2048/{freq}.npy')
    print(f'{nstd[1,0]=}')
    nstd_q = nstd[1].copy()
    nstd_u = nstd[2].copy()
    # ps = np.load('./data/ps/ps.npy')
    # noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))
    m = gen_map(beam=beam, freq=freq, lmax=lmax, rlz_idx=rlz_idx, mode='std')
    # m = gen_map(beam=beam, freq=freq, lmax=lmax, mode='std')
    m_q = m[1].copy()
    m_u = m[2].copy()
    logger.debug(f'{sys.getrefcount(m_q)-1=}')

    logger.info(f'time for fitting = {time.perf_counter()-time0}')

    df_mask = pd.read_csv(f'./mask/{freq}.csv')
    df_ps = pd.read_csv(f'../../pp_P/mask/ps_csv/{freq}.csv')

    flux_idx=2

    logger.debug(f'{sys.getrefcount(m_q)-1=}')
    obj = FitPolPS(m_q=m_q, m_u=m_u, freq=freq, nstd_q=nstd_q, nstd_u=nstd_u, flux_idx=flux_idx, df_mask=df_mask, df_ps=df_ps, lmax=lmax, nside=nside, radius_factor=1.5, beam=beam, epsilon=0.00001, threshold_extra_factor=1.5)

    # obj.calc_definite_fixed_cmb_cov()
    # obj.calc_covariance_matrix(mode='cmb+noise')

    # obj.fit_all(cov_mode='cmb+noise')
    num_ps, chi2dof, fit_P, fit_P_err, fit_phi, fit_phi_err = obj.fit_all(cov_mode='cmb+noise')


def get_ps_need_process(n_ps, threshold=3.0):
    df = pd.read_csv(f'./mask/{freq}_fit.csv')

    fit_p_list = []
    fit_p_err_list = []

    for flux_idx in range(n_ps):
        fit_q = df.at[flux_idx, 'fit_q']
        fit_q_err = df.at[flux_idx, 'fit_q_err']
        fit_u = df.at[flux_idx, 'fit_u']
        fit_u_err = df.at[flux_idx, 'fit_u_err']

        fit_p, fit_p_err = FitPolPS.calculate_P_error(Q=fit_q, U=fit_u, sigma_Q=fit_q_err, sigma_U=fit_u_err)
        print(f'{flux_idx=}, {fit_q=}, {fit_q_err=}, {fit_u=}, {fit_u_err=}, {fit_p=}, {fit_p_err=}')

        fit_p_list.append(fit_p)
        fit_p_err_list.append(fit_p_err)


    df['fit_p'] = fit_p_list
    df['fit_p_err'] = fit_p_err_list

    filtered_df = df[df['fit_p'] > threshold * df['fit_p_err']]
    # filtered_df.reset_index(drop=True)
    filtered_df.index.name = 'second_fit_index'
    filtered_df.to_csv(f'./mask/{freq}_after_filter.csv', index=True)

def second_one_ps_fit():
    rlz_idx = 0
    npix = hp.nside2npix(nside)

    time0 = time.perf_counter()
    nstd = np.load(f'../../FGSim/NSTDNORTH/2048/{freq}.npy')
    print(f'{nstd[1,0]=}')
    nstd_q = nstd[1].copy()
    nstd_u = nstd[2].copy()
    # ps = np.load('./data/ps/ps.npy')
    # noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))
    m = gen_map(beam=beam, freq=freq, lmax=lmax, rlz_idx=rlz_idx, mode='std')
    # m = gen_map(beam=beam, freq=freq, lmax=lmax, mode='std')
    m_q = m[1].copy()
    m_u = m[2].copy()
    logger.debug(f'{sys.getrefcount(m_q)-1=}')

    logger.info(f'time for fitting = {time.perf_counter()-time0}')

    df_mask = pd.read_csv(f'./mask/{freq}_after_filter.csv')
    df_ps = pd.read_csv(f'./mask/{freq}_after_filter.csv')

    for flux_idx in range(len(df_mask)):
        inv_idx = df_mask.at[flux_idx, 'second_fit_index']
        print(f'{flux_idx=}, {inv_idx=}')

        obj = FitPolPS(m_q=m_q, m_u=m_u, freq=freq, nstd_q=nstd_q, nstd_u=nstd_u, flux_idx=flux_idx, df_mask=df_mask, df_ps=df_ps, lmax=lmax, nside=nside, radius_factor=1.5, beam=beam, epsilon=0.00001, threshold_extra_factor=1.5, inv_idx=inv_idx)

        # obj.calc_definite_fixed_cmb_cov()
        # obj.calc_covariance_matrix(mode='cmb+noise')

        # obj.fit_all(cov_mode='cmb+noise')
        num_ps, chi2dof, fit_P, fit_P_err, fit_phi, fit_phi_err = obj.fit_all(cov_mode='cmb+noise')

def reduce_lists(lists):
    # Sort lists by length in descending order to prioritize longer lists
    sorted_lists = sorted(lists, key=len, reverse=True)

    # Store the final result
    result = []

    for current_list in sorted_lists:
        # Check if this list is already a subset of any list in the result
        if not any(set(current_list).issubset(set(existing_list)) for existing_list in result):
            result.append(current_list)

    return result

def second_fit_find_nearby():
    import pickle
    rlz_idx = 0
    npix = hp.nside2npix(nside)

    time0 = time.perf_counter()
    nstd = np.load(f'../../FGSim/NSTDNORTH/2048/{freq}.npy')
    print(f'{nstd[1,0]=}')
    nstd_q = nstd[1].copy()
    nstd_u = nstd[2].copy()
    # ps = np.load('./data/ps/ps.npy')
    # noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))
    m = gen_map(beam=beam, freq=freq, lmax=lmax, rlz_idx=rlz_idx, mode='std')
    # m = gen_map(beam=beam, freq=freq, lmax=lmax, mode='std')
    m_q = m[1].copy()
    m_u = m[2].copy()
    logger.debug(f'{sys.getrefcount(m_q)-1=}')

    logger.info(f'time for fitting = {time.perf_counter()-time0}')

    df_mask = pd.read_csv(f'./mask/{freq}_after_filter.csv')
    df_ps = pd.read_csv(f'./mask/{freq}_after_filter.csv')

    all_nearby_ps_list = []

    for flux_idx in range(len(df_mask)):
        inv_idx = df_mask.at[flux_idx, 'second_fit_index']
        print(f'{flux_idx=}, {inv_idx=}')

        obj = FitPolPS(m_q=m_q, m_u=m_u, freq=freq, nstd_q=nstd_q, nstd_u=nstd_u, flux_idx=flux_idx, df_mask=df_mask, df_ps=df_ps, lmax=lmax, nside=nside, radius_factor=1.5, beam=beam, epsilon=0.00001, threshold_extra_factor=1.5, inv_idx=inv_idx)

        index_list = obj.fit_all(mode='get_num_ps', cov_mode='cmb+noise')
        print(f'{index_list=}')
        all_nearby_ps_list.append(index_list)

    unique_ps_list = reduce_lists(all_nearby_ps_list)
    print(f'{unique_ps_list=}')

    with open('./mask/ps_list.pkl', 'wb') as f:
        pickle.dump(unique_ps_list, f)

def test_isinstance():
    rlz_idx = 0
    npix = hp.nside2npix(nside)

    time0 = time.perf_counter()
    nstd = np.load(f'../../FGSim/NSTDNORTH/2048/{freq}.npy')
    print(f'{nstd[1,0]=}')
    nstd_q = nstd[1].copy()
    nstd_u = nstd[2].copy()
    # ps = np.load('./data/ps/ps.npy')
    # noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))
    m = gen_map(beam=beam, freq=freq, lmax=lmax, rlz_idx=rlz_idx, mode='std')
    # m = gen_map(beam=beam, freq=freq, lmax=lmax, mode='std')
    m_q = m[1].copy()
    m_u = m[2].copy()
    logger.debug(f'{sys.getrefcount(m_q)-1=}')

    logger.info(f'time for fitting = {time.perf_counter()-time0}')

    df_mask = pd.read_csv(f'./mask/{freq}_after_filter.csv')
    df_ps = pd.read_csv(f'./mask/{freq}_after_filter.csv')

    all_nearby_ps_list = []

    flux_idx = 0
    inv_idx = df_mask.at[flux_idx, 'second_fit_index']
    print(f'{flux_idx=}, {inv_idx=}')

    # obj = FitPolPS(m_q=m_q, m_u=m_u, freq=freq, nstd_q=nstd_q, nstd_u=nstd_u, flux_idx=[14,6,8], df_mask=df_mask, df_ps=df_ps, lmax=lmax, nside=nside, radius_factor=1.5, beam=beam, epsilon=0.00001, threshold_extra_factor=1.5, inv_idx=inv_idx)

    obj = FitPolPS(m_q=m_q, m_u=m_u, freq=freq, nstd_q=nstd_q, nstd_u=nstd_u, flux_idx=[14,6,8], df_mask=df_mask, df_ps=df_ps, lmax=lmax, nside=nside, radius_factor=1.5, beam=beam, epsilon=0.00001, threshold_extra_factor=1.5, cov_path='./cmb_qu_cov_interp')
    num_ps, chi2dof, q_tuple, u_tuple = obj.fit_all(cov_mode='cmb+noise')
    print(f'{q_tuple=}, {u_tuple=}')


def second_fit_all():
    import pickle
    rlz_idx = 0
    npix = hp.nside2npix(nside)

    time0 = time.perf_counter()
    nstd = np.load(f'../../FGSim/NSTDNORTH/2048/{freq}.npy')
    print(f'{nstd[1,0]=}')
    nstd_q = nstd[1].copy()
    nstd_u = nstd[2].copy()
    # ps = np.load('./data/ps/ps.npy')
    # noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))
    # m = gen_map(beam=beam, freq=freq, lmax=lmax, rlz_idx=rlz_idx, mode='mean')
    m = gen_map(beam=beam, freq=freq, lmax=lmax, rlz_idx=rlz_idx, mode='mean', return_noise=True)
    # m = gen_map(beam=beam, freq=freq, lmax=lmax, mode='std')
    m_q = m[1].copy()
    m_u = m[2].copy()
    logger.debug(f'{sys.getrefcount(m_q)-1=}')

    logger.info(f'time for fitting = {time.perf_counter()-time0}')

    df_mask = pd.read_csv(f'./mask/{freq}_after_filter.csv')
    df_ps = pd.read_csv(f'./mask/{freq}_after_filter.csv')
    with open('./mask/ps_list.pkl', 'rb') as f:
        ps_list = pickle.load(f)

    n_ps = len(df_mask)

    flux_idx_arr = df_mask.loc[:n_ps-1, 'flux_idx'].to_numpy()
    index_arr = df_mask.loc[:n_ps-1, 'index'].to_numpy()
    lon_arr = df_mask.loc[:n_ps-1, 'lon'].to_numpy()
    lat_arr = df_mask.loc[:n_ps-1, 'lat'].to_numpy()

    num_ps_arr = np.zeros_like(flux_idx_arr, dtype=float)
    chi2dof_arr = np.zeros_like(flux_idx_arr, dtype=float)
    true_q_arr = np.zeros_like(flux_idx_arr, dtype=float)
    fit_q_arr = np.zeros_like(flux_idx_arr, dtype=float)
    true_u_arr = np.zeros_like(flux_idx_arr, dtype=float)
    fit_u_arr = np.zeros_like(flux_idx_arr, dtype=float)


    for idx in ps_list:
        print(f'{idx=}')
        if len(idx) == 1:
            flux_idx = idx[0]
            inv_idx = df_mask.at[flux_idx, 'second_fit_index']
            print(f'{flux_idx=}, {inv_idx=}')

            obj = FitPolPS(m_q=m_q, m_u=m_u, freq=freq, nstd_q=nstd_q, nstd_u=nstd_u, flux_idx=flux_idx, df_mask=df_mask, df_ps=df_ps, lmax=lmax, nside=nside, radius_factor=1.5, beam=beam, epsilon=0.00001, threshold_extra_factor=1.5, inv_idx=inv_idx, cov_path='./cmb_qu_cov_interp')
            # num_ps, chi2dof, fit_P, fit_P_err, fit_phi, fit_phi_err = obj.fit_all(cov_mode='cmb+noise')
            num_ps, chi2dof, true_q, fit_q, fit_q_err, true_u, fit_u, fit_u_err = obj.fit_all(cov_mode='cmb+noise', return_qu=True)
            num_ps_arr[flux_idx] = num_ps
            chi2dof_arr[flux_idx] = chi2dof
            true_q_arr[flux_idx] = true_q
            fit_q_arr[flux_idx] = fit_q
            true_u_arr[flux_idx] = true_u
            fit_u_arr[flux_idx] = fit_u

        else:
            flux_idx = idx
            obj = FitPolPS(m_q=m_q, m_u=m_u, freq=freq, nstd_q=nstd_q, nstd_u=nstd_u, flux_idx=flux_idx, df_mask=df_mask, df_ps=df_ps, lmax=lmax, nside=nside, radius_factor=1.5, beam=beam, epsilon=0.00001, threshold_extra_factor=1.5, cov_path='./cmb_qu_cov_interp')
            # obj.calc_definite_fixed_cmb_cov()
            # obj.calc_covariance_matrix(mode='cmb+noise')
            # num_ps, chi2dof, fit_P, fit_P_err, fit_phi, fit_phi_err = obj.fit_all(cov_mode='cmb+noise')
            num_ps, chi2dof, q_tuple, u_tuple = obj.fit_all(cov_mode='cmb+noise')
            print(f'{q_tuple=}, {u_tuple=}')

            for i, (fit_q, fit_u) in enumerate(zip(q_tuple, u_tuple)):
                _idx = flux_idx[i]
                true_q = FitPolPS.mJy_to_uKCMB(intensity_mJy=df_mask.at[flux_idx[i], 'qflux'], frequency_GHz=freq) / hp.nside2pixarea(nside=nside)
                true_u = FitPolPS.mJy_to_uKCMB(intensity_mJy=df_mask.at[flux_idx[i], 'uflux'], frequency_GHz=freq) / hp.nside2pixarea(nside=nside)
                print(f'flux_idx = {flux_idx[i]}, {fit_q=}, {fit_u=}, {true_q=}, {true_u=}')
                num_ps_arr[_idx] = num_ps
                chi2dof_arr[_idx] = chi2dof
                true_q_arr[_idx] = true_q
                fit_q_arr[_idx] = fit_q
                true_u_arr[_idx] = true_u
                fit_u_arr[_idx] = fit_u

    df_fit = pd.DataFrame({
        'flux_idx': flux_idx_arr,
        'index': index_arr,
        'lon': lon_arr,
        'lat': lat_arr,
        'num_ps': num_ps_arr,
        'chi2dof': chi2dof_arr,
        'true_q': true_q_arr,
        'fit_q': fit_q_arr,
        'true_u': true_u_arr,
        'fit_u': fit_u_arr,
        })

    path_csv = Path('./mask/noise')
    path_csv.mkdir(exist_ok=True, parents=True)
    df_fit.to_csv(f'./mask/noise/{rlz_idx}.csv', index=False)

def do_eblc():
    rlz_idx = 0
    def calc_eblc_res(sim_mode):
        pcfn, cfn, cf, n = gen_map_all(beam=beam, freq=freq, lmax=lmax, rlz_idx=rlz_idx, mode=sim_mode)
        mask = hp.read_map('./inpainting/mask/mask_only_edge.fits')

        obj_pcfn = EBLeakageCorrection(m=pcfn, lmax=lmax, nside=nside, mask=mask, post_mask=mask)
        _, _, cln_pcfn = obj_pcfn.run_eblc()
        slope = obj_pcfn.return_slope()

        obj_cfn = EBLeakageCorrection(m=cfn, lmax=lmax, nside=nside, mask=mask, post_mask=mask, slope_in=slope)
        _, _, cln_cfn = obj_cfn.run_eblc()

        obj_cf = EBLeakageCorrection(m=cf, lmax=lmax, nside=nside, mask=mask, post_mask=mask, slope_in=slope)
        _, _, cln_cf = obj_cf.run_eblc()

        obj_n = EBLeakageCorrection(m=n, lmax=lmax, nside=nside, mask=mask, post_mask=mask, slope_in=slope)
        _, _, cln_n = obj_n.run_eblc()

        rmv_q = np.load(f'./fit_res/{sim_mode}/3sigma/map_q_{rlz_idx}.npy')
        rmv_u = np.load(f'./fit_res/{sim_mode}/3sigma/map_u_{rlz_idx}.npy')
        rmv_t = np.zeros_like(rmv_q)

        obj_rmv = EBLeakageCorrection(m=np.asarray([rmv_t, rmv_q, rmv_u]), lmax=lmax, nside=nside, mask=mask, post_mask=mask, slope_in=slope)
        _, _, cln_rmv = obj_rmv.run_eblc()

        path_eblc_pcfn = Path(f'./fit_res/{sim_mode}/pcfn')
        path_eblc_pcfn.mkdir(exist_ok=True, parents=True)
        np.save(path_eblc_pcfn / Path(f'{rlz_idx}.npy'), cln_pcfn)

        path_eblc_cfn = Path(f'./fit_res/{sim_mode}/cfn')
        path_eblc_cfn.mkdir(exist_ok=True, parents=True)
        np.save(path_eblc_cfn / Path(f'{rlz_idx}.npy'), cln_cfn)

        path_eblc_cf = Path(f'./fit_res/{sim_mode}/cf')
        path_eblc_cf.mkdir(exist_ok=True, parents=True)
        np.save(path_eblc_cf / Path(f'{rlz_idx}.npy'), cln_cf)

        path_eblc_n = Path(f'./fit_res/{sim_mode}/n')
        path_eblc_n.mkdir(exist_ok=True, parents=True)
        np.save(path_eblc_n / Path(f'{rlz_idx}.npy'), cln_n)

        path_eblc_rmv = Path(f'./fit_res/{sim_mode}/rmv')
        path_eblc_rmv.mkdir(exist_ok=True, parents=True)
        np.save(path_eblc_rmv / Path(f'{rlz_idx}.npy'), cln_rmv)

    calc_eblc_res(sim_mode='mean')
    calc_eblc_res(sim_mode='std')

def do_eblc_n():
    rlz_idx = 0
    def calc_eblc_res(sim_mode):
        pcfn, cfn, cf, n = gen_map_all(beam=beam, freq=freq, lmax=lmax, rlz_idx=rlz_idx, mode=sim_mode)
        mask = hp.read_map('./inpainting/mask/mask_only_edge.fits')

        obj_pcfn = EBLeakageCorrection(m=pcfn, lmax=lmax, nside=nside, mask=mask, post_mask=mask)
        _, _, cln_pcfn = obj_pcfn.run_eblc()
        slope = obj_pcfn.return_slope()

        rmv_q = np.load(f'./fit_res/noise/3sigma/map_q_{rlz_idx}.npy')
        rmv_u = np.load(f'./fit_res/noise/3sigma/map_u_{rlz_idx}.npy')
        rmv_t = np.zeros_like(rmv_q)

        obj_rmv = EBLeakageCorrection(m=np.asarray([rmv_t, rmv_q, rmv_u]), lmax=lmax, nside=nside, mask=mask, post_mask=mask, slope_in=slope)
        _, _, cln_rmv = obj_rmv.run_eblc()

        path_eblc_rmv = Path(f'./fit_res/noise/rmv')
        path_eblc_rmv.mkdir(exist_ok=True, parents=True)
        np.save(path_eblc_rmv / Path(f'{rlz_idx}.npy'), cln_rmv)

    calc_eblc_res(sim_mode='mean')
    # calc_eblc_res(sim_mode='std')

def check_eblc_res():
    rlz_idx = 0
    sim_mode = 'mean'

    df = pd.read_csv(f'./mask/{freq}_after_filter.csv')

    cln_pcfn = np.load(f'./fit_res/{sim_mode}/pcfn/{rlz_idx}.npy')
    cln_cfn = np.load(f'./fit_res/{sim_mode}/cfn/{rlz_idx}.npy')
    # cln_cf = np.load(f'./fit_res/{sim_mode}/pcfn/{rlz_idx}.npy')
    cln_rmv = np.load(f'./fit_res/{sim_mode}/rmv/{rlz_idx}.npy')
    inp = hp.read_map(f'./inpainting/output_m2_{sim_mode}/{rlz_idx}.fits')
    cln_n = np.load(f'./fit_res/mean/n/{rlz_idx}.npy')
    rmv_n = np.load(f'./fit_res/mean/n/{rlz_idx}.npy')
    inp_n = hp.read_map(f'./inpainting/output_m2_n/{rlz_idx}.fits')

    hp.orthview(cln_pcfn, rot=[100,50,0], title='pcfn', half_sky=True)
    hp.orthview(cln_cfn, rot=[100,50,0], title='cfn', half_sky=True)
    hp.orthview(cln_rmv, rot=[100,50,0], title='rmv', half_sky=True)
    hp.orthview(inp, rot=[100,50,0], title='inp', half_sky=True)
    hp.orthview(cln_n, rot=[100,50,0], title='cln n', half_sky=True)
    hp.orthview(rmv_n, rot=[100,50,0], title='rmv n', half_sky=True)
    hp.orthview(inp_n, rot=[100,50,0], title='inp n', half_sky=True)
    plt.show()

    for flux_idx in np.arange(len(df)):
        lon = np.rad2deg(df.at[flux_idx, 'lon'])
        lat = np.rad2deg(df.at[flux_idx, 'lat'])
        hp.gnomview(cln_pcfn, rot=[lon, lat, 0], title='pcfn')
        hp.gnomview(cln_cfn, rot=[lon, lat, 0], title='cfn')
        hp.gnomview(cln_rmv, rot=[lon, lat, 0], title='rmv')
        hp.gnomview(inp, rot=[lon, lat, 0], title='inp')
        hp.gnomview(cln_n, rot=[lon, lat, 0], title='cln n')
        hp.gnomview(rmv_n, rot=[lon, lat, 0], title='rmv n')
        hp.gnomview(inp_n, rot=[lon, lat, 0], title='inp n')
        plt.show()

def smooth_map(map_in, mask, lmax, beam_in, beam_out):
    bl_in = hp.gauss_beam(fwhm=np.deg2rad(beam_in)/60, lmax=lmax, pol=True)[:,2]
    bl_out = hp.gauss_beam(fwhm=np.deg2rad(beam_out)/60, lmax=lmax, pol=True)[:,2]
    print(f'{bl_in.shape=}')
    # alm_, rel_res, n_iter = hp.map2alm_lsq(map_in*mask, lmax=lmax, mmax=lmax, tol=1e-15)
    alm_ = hp.map2alm(map_in*mask, lmax=lmax)
    # print(f'{rel_res=}, {n_iter=}')
    map_out = hp.alm2map(hp.almxfl(alm_, fl=bl_out/bl_in), nside=nside)
    return map_out

def smooth_all():
    rlz_idx = 0
    beam_out = 17
    mask = np.load('../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5APO_3.npy')
    # mask_check = np.load('../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5APO_3APO_5.npy')
    def calc_smooth(sim_mode):
        if sim_mode == 'noise':
            n = np.load(f'./fit_res/mean/n/{rlz_idx}.npy')
            n_inp = hp.read_map(f'./inpainting/output_m2_n/{rlz_idx}.fits')
            n_rmv = np.load(f'./fit_res/noise/rmv/{rlz_idx}.npy')

            sm_n = smooth_map(map_in=n, mask=mask, lmax=lmax, beam_in=beam, beam_out=beam_out)
            sm_n_inp = smooth_map(map_in=n_inp, mask=mask, lmax=lmax, beam_in=beam, beam_out=beam_out)
            sm_n_rmv = smooth_map(map_in=n_rmv, mask=mask, lmax=lmax, beam_in=beam, beam_out=beam_out)

            path_n = Path(f'./fit_res/sm/{sim_mode}/n')
            path_n_rmv = Path(f'./fit_res/sm/{sim_mode}/n_rmv')
            path_n_inp = Path(f'./fit_res/sm/{sim_mode}/n_inp')

            path_n.mkdir(exist_ok=True, parents=True)
            path_n_rmv.mkdir(exist_ok=True, parents=True)
            path_n_inp.mkdir(exist_ok=True, parents=True)
            np.save(path_n / Path(f'{rlz_idx}.npy'), sm_n)
            np.save(path_n_inp / Path(f'{rlz_idx}.npy'), sm_n_inp)
            np.save(path_n_rmv / Path(f'{rlz_idx}.npy'), sm_n_rmv)
        else:
            pcfn = np.load(f'./fit_res/{sim_mode}/pcfn/{rlz_idx}.npy')
            cfn = np.load(f'./fit_res/{sim_mode}/cfn/{rlz_idx}.npy')
            cf = np.load(f'./fit_res/{sim_mode}/cf/{rlz_idx}.npy')
            rmv = np.load(f'./fit_res/{sim_mode}/rmv/{rlz_idx}.npy')
            inp = hp.read_map(f'./inpainting/output_m2_{sim_mode}/{rlz_idx}.fits')
            sm_pcfn = smooth_map(map_in=pcfn, mask=mask, lmax=lmax, beam_in=beam, beam_out=beam_out)
            sm_cfn = smooth_map(map_in=cfn, mask=mask, lmax=lmax, beam_in=beam, beam_out=beam_out)
            sm_cf = smooth_map(map_in=cf, mask=mask, lmax=lmax, beam_in=beam, beam_out=beam_out)
            sm_rmv = smooth_map(map_in=rmv, mask=mask, lmax=lmax, beam_in=beam, beam_out=beam_out)
            sm_inp = smooth_map(map_in=inp, mask=mask, lmax=lmax, beam_in=beam, beam_out=beam_out)
            path_pcfn = Path(f'./fit_res/sm/{sim_mode}/pcfn')
            path_cfn = Path(f'./fit_res/sm/{sim_mode}/cfn')
            path_cf = Path(f'./fit_res/sm/{sim_mode}/cf')
            path_rmv = Path(f'./fit_res/sm/{sim_mode}/rmv')
            path_inp = Path(f'./fit_res/sm/{sim_mode}/inp')
            path_pcfn.mkdir(exist_ok=True, parents=True)
            path_cfn.mkdir(exist_ok=True, parents=True)
            path_cf.mkdir(exist_ok=True, parents=True)
            path_rmv.mkdir(exist_ok=True, parents=True)
            path_inp.mkdir(exist_ok=True, parents=True)

            np.save(path_pcfn / Path(f'{rlz_idx}.npy'), sm_pcfn)
            np.save(path_cfn / Path(f'{rlz_idx}.npy'), sm_cfn)
            np.save(path_cf / Path(f'{rlz_idx}.npy'), sm_cf)
            np.save(path_rmv / Path(f'{rlz_idx}.npy'), sm_rmv)
            np.save(path_inp / Path(f'{rlz_idx}.npy'), sm_inp)
    calc_smooth(sim_mode='mean')
    calc_smooth(sim_mode='std')
    calc_smooth(sim_mode='noise')

def smooth_check_all():
    rlz_idx = 0
    beam_out = 17
    bl_in = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax, pol=True)[:,2]
    bl_out = hp.gauss_beam(fwhm=np.deg2rad(beam_out)/60, lmax=lmax, pol=True)[:,2]
    sim_mode = 'mean'
    mask_check = np.load('../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5APO_3APO_5.npy')

    pcfn = np.load(f'./fit_res/{sim_mode}/pcfn/{rlz_idx}.npy')
    cfn = np.load(f'./fit_res/{sim_mode}/cfn/{rlz_idx}.npy')
    cf = np.load(f'./fit_res/{sim_mode}/cf/{rlz_idx}.npy')
    rmv = np.load(f'./fit_res/{sim_mode}/rmv/{rlz_idx}.npy')
    inp = hp.read_map(f'./inpainting/output_m2_{sim_mode}/{rlz_idx}.fits')

    sm_pcfn = np.load(f'./fit_res/sm/{sim_mode}/pcfn/{rlz_idx}.npy')
    sm_cfn = np.load(f'./fit_res/sm/{sim_mode}/cfn/{rlz_idx}.npy')
    sm_cf = np.load(f'./fit_res/sm/{sim_mode}/cf/{rlz_idx}.npy')
    sm_rmv = np.load(f'./fit_res/sm/{sim_mode}/rmv/{rlz_idx}.npy')
    sm_inp = np.load(f'./fit_res/sm/{sim_mode}/inp/{rlz_idx}.npy')

    n = np.load(f'./fit_res/mean/n/{rlz_idx}.npy')
    n_inp = hp.read_map(f'./inpainting/output_m2_n/{rlz_idx}.fits')
    n_rmv = np.load(f'./fit_res/noise/rmv/{rlz_idx}.npy')
    sm_n = np.load(f'./fit_res/sm/noise/n/{rlz_idx}.npy')
    sm_n_inp = np.load(f'./fit_res/sm/noise/n_inp/{rlz_idx}.npy')
    sm_n_rmv = np.load(f'./fit_res/sm/noise/n_rmv/{rlz_idx}.npy')

    cl_pcfn = hp.anafast(pcfn*mask_check, lmax=lmax)
    cl_sm_pcfn = hp.anafast(sm_pcfn*mask_check, lmax=lmax)
    plt.loglog(cl_pcfn/bl_in**2, label='pcfn in')
    plt.loglog(cl_sm_pcfn/bl_out**2, label='pcfn out')
    plt.legend()
    plt.show()

    cl_cfn = hp.anafast(cfn*mask_check, lmax=lmax)
    cl_sm_cfn = hp.anafast(sm_cfn*mask_check, lmax=lmax)
    plt.loglog(cl_cfn/bl_in**2, label='cfn in')
    plt.loglog(cl_sm_cfn/bl_out**2, label='cfn out')
    plt.legend()
    plt.show()

    cl_cf = hp.anafast(cf*mask_check, lmax=lmax)
    cl_sm_cf = hp.anafast(sm_cf*mask_check, lmax=lmax)
    plt.loglog(cl_cf/bl_in**2, label='cf in')
    plt.loglog(cl_sm_cf/bl_out**2, label='cf out')
    plt.legend()
    plt.show()

    cl_rmv = hp.anafast(rmv*mask_check, lmax=lmax)
    cl_sm_rmv = hp.anafast(sm_rmv*mask_check, lmax=lmax)
    plt.loglog(cl_rmv/bl_in**2, label='rmv in')
    plt.loglog(cl_sm_rmv/bl_out**2, label='rmv out')
    plt.legend()
    plt.show()

    cl_inp = hp.anafast(inp*mask_check, lmax=lmax)
    cl_sm_inp = hp.anafast(sm_inp*mask_check, lmax=lmax)
    plt.loglog(cl_inp/bl_in**2, label='inp in')
    plt.loglog(cl_sm_inp/bl_out**2, label='inp out')
    plt.legend()
    plt.show()

    cl_n = hp.anafast(n*mask_check, lmax=lmax)
    cl_sm_n = hp.anafast(sm_n*mask_check, lmax=lmax)
    plt.loglog(cl_n/bl_in**2, label='n in')
    plt.loglog(cl_sm_n/bl_out**2, label='n out')
    plt.legend()
    plt.show()
    cl_n_inp = hp.anafast(n_inp*mask_check, lmax=lmax)
    cl_sm_n_inp = hp.anafast(sm_n_inp*mask_check, lmax=lmax)
    plt.loglog(cl_n_inp/bl_in**2, label='n_inp in')
    plt.loglog(cl_sm_n_inp/bl_out**2, label='n_inp out')
    plt.legend()
    plt.show()
    cl_n_rmv = hp.anafast(n_rmv*mask_check, lmax=lmax)
    cl_sm_n_rmv = hp.anafast(sm_n_rmv*mask_check, lmax=lmax)
    plt.loglog(cl_n_rmv/bl_in**2, label='n_rmv in')
    plt.loglog(cl_sm_n_rmv/bl_out**2, label='n_rmv out')
    plt.legend()
    plt.show()

def smooth_check():
    mask = np.load('../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5APO_3.npy')
    mask_check = np.load('../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5APO_3APO_5.npy')
    map_in = np.load(f'./fit_res/mean/rmv/0.npy')

    beam_out = 17
    bl_in = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax, pol=True)[:,2]
    bl_out = hp.gauss_beam(fwhm=np.deg2rad(beam_out)/60, lmax=lmax, pol=True)[:,2]
    map_out = smooth_map(map_in=map_in, mask=mask, lmax=lmax, beam_in=beam, beam_out=beam_out)
    # map_re_out = smooth_map(map_in=map_out, mask=mask, lmax=lmax, beam_in=17, beam_out=67)
    # map_deproj = hp.alm2map(hp.map2alm(map_out, lmax=300), nside=nside)

    hp.orthview(map_in, rot=[100,50,0], half_sky=True, min=-15, max=15)
    hp.orthview(map_out, rot=[100,50,0], half_sky=True, min=-15, max=15)
    # hp.orthview(map_deproj, rot=[100,50,0], half_sky=True, min=-15, max=15)
    plt.show()

    cl_in = hp.anafast(map_in*mask_check, lmax=lmax)
    cl_out = hp.anafast(map_out*mask_check, lmax=lmax)
    # cl_re_out = hp.anafast(map_re_out*mask_check, lmax=lmax)
    plt.loglog(cl_in/bl_in**2, label='in')
    plt.loglog(cl_out/bl_out**2, label='out')
    # plt.loglog(cl_re_out/bl_in**2, label='re out')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # gen_pix_idx(flux_idx=0)
    # gen_cov_inv()
    # check_parameter_distribution()
    # first_fit_all()
    # one_ps_fit()
    # get_ps_need_process(n_ps=ps_number)

    # second_one_ps_fit()
    # second_fit_find_nearby()
    # second_fit_all()

    # test_isinstance()
    # do_eblc()
    # do_eblc_n()
    # check_eblc_res()

    # smooth_all()
    # smooth_check()
    smooth_check_all()
    pass


