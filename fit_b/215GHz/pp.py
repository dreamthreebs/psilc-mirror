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
    # save_path = Path(f'fit_res/params/mean_for_sigma')
    # save_path.mkdir(exist_ok=True, parents=True)

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

def find_connected_components(lists):
    # Convert lists to sets for easier manipulation
    sets = [set(lst) for lst in lists]

    # Keep track of which sets have been merged
    merged = []
    used = set()

    # Check each set against other sets
    for i, set1 in enumerate(sets):
        if i in used:
            continue

        current = set1.copy()
        used.add(i)

        # Keep checking for connections until no more are found
        changed = True
        while changed:
            changed = False
            for j, set2 in enumerate(sets):
                if j in used:
                    continue

                # If sets share any elements, merge them
                if current & set2:
                    current |= set2
                    used.add(j)
                    changed = True

        merged.append(sorted(list(current)))

    return merged


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

    unique_ps_list = find_connected_components(all_nearby_ps_list)
    print(f'{unique_ps_list=}')

    with open('./mask/ps_list.pkl', 'wb') as f:
        pickle.dump(unique_ps_list, f)

def second_gen_pix_and_inv():
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
    with open('./mask/ps_list.pkl', 'rb') as f:
        ps_list = pickle.load(f)

    for idx in ps_list:
        print(f'{idx=}')
        if len(idx) == 1:
            continue
        else:
            obj = FitPolPS(m_q=m_q, m_u=m_u, freq=freq, nstd_q=nstd_q, nstd_u=nstd_u, flux_idx=idx, df_mask=df_mask, df_ps=df_ps, lmax=lmax, nside=nside, radius_factor=1.5, beam=beam, epsilon=0.00001, threshold_extra_factor=1.5, cov_path='./cmb_qu_cov_interp')

            obj.calc_definite_fixed_cmb_cov()
            obj.calc_covariance_matrix(mode='cmb+noise')

            # num_ps, chi2dof, q_tuple, u_tuple = obj.fit_all(cov_mode='cmb+noise')
            # print(f'{q_tuple=}, {u_tuple=}')


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
    mode = 'mean'
    if mode == "noise":
        m = gen_map(beam=beam, freq=freq, lmax=lmax, rlz_idx=rlz_idx, mode='mean', return_noise=True)
    else:
        m = gen_map(beam=beam, freq=freq, lmax=lmax, rlz_idx=rlz_idx, mode=mode)
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

    path_csv = Path(f'./mask/{mode}')
    path_csv.mkdir(exist_ok=True, parents=True)
    df_fit.to_csv(path_csv / Path(f'{rlz_idx}.csv'), index=False)



if __name__ == '__main__':
    # gen_pix_idx(flux_idx=0)
    # gen_cov_inv()
    # check_parameter_distribution()
    # first_fit_all()
    # one_ps_fit()
    # get_ps_need_process(n_ps=ps_number)

    # second_one_ps_fit()
    # second_fit_find_nearby()
    # second_gen_pix_and_inv()
    second_fit_all()

    pass



