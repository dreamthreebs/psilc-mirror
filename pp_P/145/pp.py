import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pandas as pd
import time
import pickle
import os,sys
import logging
import ipdb

from pathlib import Path
from iminuit import Minuit
from iminuit.cost import LeastSquares

from fit_qu_base import FitPolPS

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.INFO)

freq = 145
lmax = 1999
nside = 2048
beam = 19
n_flux_idx = 150

def filter_df():
    # filter pandas dataframe where flux bigger than some threshold
    df_mask = pd.read_csv(f'../mask/mask_csv/{freq}.csv')
    filtered_df = df_mask[df_mask['pflux']>1]
    logger.info(f'{filtered_df=}')
    logger.info(f'{len(filtered_df)=}')

def calc_number():
    # calculate number of point source near one point source

    time0 = time.perf_counter()
    # m = np.load(f'../../fitdata/synthesis_data/2048/PSNOISE/{freq}/0.npy')
    m = np.load(f'../../fitdata/synthesis_data/2048/PSCMBNOISE/{freq}/2.npy')
    m_q = m[1].copy()
    m_u = m[2].copy()
    logger.debug(f'{sys.getrefcount(m_q)-1=}')

    logger.info(f'time for fitting = {time.perf_counter()-time0}')
    nstd = np.load(f'../../FGSim/NSTDNORTH/2048/{freq}.npy')
    nstd_q = nstd[1].copy()
    nstd_u = nstd[2].copy()
    df_mask = pd.read_csv(f'../mask/mask_csv/{freq}.csv')
    df_ps = pd.read_csv(f'../mask/ps_csv/{freq}.csv')

    num_ps_list = []
    overlap_list = []
    for flux_idx in range(700):

        logger.debug(f'{sys.getrefcount(m_q)-1=}')
        obj = FitPolPS(m_q=m_q, m_u=m_u, freq=freq, nstd_q=nstd_q, nstd_u=nstd_u, flux_idx=flux_idx, df_mask=df_mask, df_ps=df_ps, lmax=lmax, nside=nside, radius_factor=1.5, beam=beam, epsilon=0.00001)

        logger.debug(f'{sys.getrefcount(m_q)-1=}')
        # obj.see_true_map(m_q=m_q, m_u=m_u, nside=nside, beam=beam)

        num_ps, num_near_ps, ang_near = obj.fit_all(cov_mode='noise', mode='get_num_ps')
        overlap_flag = obj.fit_all(cov_mode='noise', mode='get_overlap')
        logger.debug(f'{num_ps=}, {num_near_ps=}, {ang_near}, {overlap_flag}')
        num_ps_list.append(num_ps)
        if overlap_flag == True:
            overlap_list.append(flux_idx)

    num_ps_arr = np.asarray(num_ps_list)
    overlap_arr = np.asarray(overlap_list)
    print(f'{num_ps_arr=}')
    print(f'{overlap_arr=}')
    np.save('./num_ps.npy', num_ps_arr)
    np.save('./overlap_ps.npy', overlap_arr)

def check_num_ps_overlap():
    num_ps = np.load('./num_ps.npy')
    overlap_ps = np.load('./overlap_ps.npy')
    print(f'{num_ps=}')
    print(f'{overlap_ps=}')

def get_disc_pix_ind():
    # get pixel index for every point sources in catelogue

    time0 = time.perf_counter()
    # m = np.load(f'../../fitdata/synthesis_data/2048/PSNOISE/{freq}/0.npy')
    m = np.load(f'../../fitdata/synthesis_data/2048/PSCMBNOISE/{freq}/2.npy')
    m_q = m[1].copy()
    m_u = m[2].copy()
    logger.debug(f'{sys.getrefcount(m_q)-1=}')

    logger.info(f'time for fitting = {time.perf_counter()-time0}')
    nstd = np.load(f'../../FGSim/NSTDNORTH/2048/{freq}.npy')
    nstd_q = nstd[1].copy()
    nstd_u = nstd[2].copy()
    df_mask = pd.read_csv(f'../mask/mask_csv/{freq}.csv')
    df_ps = pd.read_csv(f'../mask/ps_csv/{freq}.csv')

    for flux_idx in range(n_flux_idx):
        obj = FitPolPS(m_q=m_q, m_u=m_u, freq=freq, nstd_q=nstd_q, nstd_u=nstd_u, flux_idx=flux_idx, df_mask=df_mask, df_ps=df_ps, lmax=lmax, nside=nside, radius_factor=1.5, beam=beam, epsilon=0.00001)
        pix_ind = obj.get_pix_ind()
        path_pix_ind = Path('./pix_idx')
        path_pix_ind.mkdir(exist_ok=True, parents=True)
        np.save(path_pix_ind / Path(f'./{flux_idx}.npy'), pix_ind)
        print(f'{pix_ind=}')

def calc_cov():
    # get pixel index for every point sources in catelogue

    time0 = time.perf_counter()
    # m = np.load(f'../../fitdata/synthesis_data/2048/PSNOISE/{freq}/0.npy')
    m = np.load(f'../../fitdata/synthesis_data/2048/PSCMBNOISE/{freq}/2.npy')
    m_q = m[1].copy()
    m_u = m[2].copy()
    logger.debug(f'{sys.getrefcount(m_q)-1=}')

    logger.info(f'time for fitting = {time.perf_counter()-time0}')
    nstd = np.load(f'../../FGSim/NSTDNORTH/2048/{freq}.npy')
    nstd_q = nstd[1].copy()
    nstd_u = nstd[2].copy()
    df_mask = pd.read_csv(f'../mask/mask_csv/{freq}.csv')
    df_ps = pd.read_csv(f'../mask/ps_csv/{freq}.csv')

    for flux_idx in range(n_flux_idx):
        obj = FitPolPS(m_q=m_q, m_u=m_u, freq=freq, nstd_q=nstd_q, nstd_u=nstd_u, flux_idx=flux_idx, df_mask=df_mask, df_ps=df_ps, lmax=lmax, nside=nside, radius_factor=1.5, beam=beam, epsilon=0.00001)
        obj.calc_definite_fixed_cmb_cov()
        obj.calc_covariance_matrix(mode='cmb+noise')

def save_fit_res_to_csv():

    for rlz_idx in range(100):
        print(f'{rlz_idx=}')
        df_mask = pd.read_csv(f'../mask/mask_csv/{freq}.csv')
        q_amp_true = FitPolPS.mJy_to_uKCMB(1, freq) * df_mask["qflux"].to_numpy()
        u_amp_true = FitPolPS.mJy_to_uKCMB(1, freq) * df_mask["uflux"].to_numpy()

        num_ps_arr = np.zeros(700)
        q_amp_arr = np.zeros(700)
        q_amp_err_arr = np.zeros(700)
        u_amp_arr = np.zeros(700)
        u_amp_err_arr = np.zeros(700)
        chi2dof_arr = np.zeros(700)
        fit_err_q_arr = np.zeros(700)
        fit_err_u_arr = np.zeros(700)
        for flux_idx in range(150):
            if flux_idx in [199,]:
                continue
            print(f'{flux_idx=}')
            num_ps_arr[flux_idx] = np.load(f'./fit_res/2048/PSCMBNOISE/1.5/idx_{flux_idx}/num_ps.npy')[rlz_idx]
            q_amp_arr[flux_idx] = np.load(f'./fit_res/2048/PSCMBNOISE/1.5/idx_{flux_idx}/q_amp.npy')[rlz_idx]
            q_amp_err_arr[flux_idx] = np.load(f'./fit_res/2048/PSCMBNOISE/1.5/idx_{flux_idx}/q_amp_err.npy')[rlz_idx]
            u_amp_arr[flux_idx] = np.load(f'./fit_res/2048/PSCMBNOISE/1.5/idx_{flux_idx}/u_amp.npy')[rlz_idx]
            u_amp_err_arr[flux_idx] = np.load(f'./fit_res/2048/PSCMBNOISE/1.5/idx_{flux_idx}/u_amp_err.npy')[rlz_idx]
            chi2dof_arr[flux_idx] = np.load(f'./fit_res/2048/PSCMBNOISE/1.5/idx_{flux_idx}/chi2dof.npy')[rlz_idx]
            fit_err_q_arr[flux_idx] = np.load(f'./fit_res/2048/PSCMBNOISE/1.5/idx_{flux_idx}/fit_error_q.npy')[rlz_idx]
            fit_err_u_arr[flux_idx] = np.load(f'./fit_res/2048/PSCMBNOISE/1.5/idx_{flux_idx}/fit_error_u.npy')[rlz_idx]

        df_mask['q_amp_true'] = q_amp_true
        df_mask['q_amp_fit'] = q_amp_arr
        df_mask['q_amp_err'] = q_amp_err_arr
        df_mask['fit_error_q'] = fit_err_q_arr

        df_mask['u_amp_true'] = u_amp_true
        df_mask['u_amp_fit'] = u_amp_arr
        df_mask['u_amp_err'] = u_amp_err_arr
        df_mask['fit_error_u'] = fit_err_u_arr

        df_mask['chi2dof'] = chi2dof_arr
        df_mask['num_ps'] = num_ps_arr
        path_csv = Path('./fit_res/2048/PSCMBNOISE/csv')
        path_csv.mkdir(parents=True, exist_ok=True)
        df_mask.to_csv(path_csv / Path(f'{rlz_idx}.csv'), index=False)

def save_pcfn_fit_res_to_csv():

    for rlz_idx in range(100):
        print(f'{rlz_idx=}')
        df_mask = pd.read_csv(f'../mask/mask_csv/{freq}.csv')
        q_amp_true = FitPolPS.mJy_to_uKCMB(1, freq) * df_mask["qflux"].to_numpy()
        u_amp_true = FitPolPS.mJy_to_uKCMB(1, freq) * df_mask["uflux"].to_numpy()

        num_ps_arr = np.zeros(700)
        q_amp_arr = np.zeros(700)
        q_amp_err_arr = np.zeros(700)
        u_amp_arr = np.zeros(700)
        u_amp_err_arr = np.zeros(700)
        chi2dof_arr = np.zeros(700)
        fit_err_q_arr = np.zeros(700)
        fit_err_u_arr = np.zeros(700)
        for flux_idx in range(n_flux_idx):

            # if flux_idx in None:
            #     continue

            print(f'{flux_idx=}')
            num_ps_arr[flux_idx] = np.load(f'./fit_res/2048/PSCMBFGNOISE/1.5/idx_{flux_idx}/num_ps.npy')[rlz_idx]
            q_amp_arr[flux_idx] = np.load(f'./fit_res/2048/PSCMBFGNOISE/1.5/idx_{flux_idx}/q_amp.npy')[rlz_idx]
            q_amp_err_arr[flux_idx] = np.load(f'./fit_res/2048/PSCMBFGNOISE/1.5/idx_{flux_idx}/q_amp_err.npy')[rlz_idx]
            u_amp_arr[flux_idx] = np.load(f'./fit_res/2048/PSCMBFGNOISE/1.5/idx_{flux_idx}/u_amp.npy')[rlz_idx]
            u_amp_err_arr[flux_idx] = np.load(f'./fit_res/2048/PSCMBFGNOISE/1.5/idx_{flux_idx}/u_amp_err.npy')[rlz_idx]
            chi2dof_arr[flux_idx] = np.load(f'./fit_res/2048/PSCMBFGNOISE/1.5/idx_{flux_idx}/chi2dof.npy')[rlz_idx]
            fit_err_q_arr[flux_idx] = np.load(f'./fit_res/2048/PSCMBFGNOISE/1.5/idx_{flux_idx}/fit_error_q.npy')[rlz_idx]
            fit_err_u_arr[flux_idx] = np.load(f'./fit_res/2048/PSCMBFGNOISE/1.5/idx_{flux_idx}/fit_error_u.npy')[rlz_idx]

        df_mask['q_amp_true'] = q_amp_true
        df_mask['q_amp_fit'] = q_amp_arr
        df_mask['q_amp_err'] = q_amp_err_arr
        df_mask['fit_error_q'] = fit_err_q_arr

        df_mask['u_amp_true'] = u_amp_true
        df_mask['u_amp_fit'] = u_amp_arr
        df_mask['u_amp_err'] = u_amp_err_arr
        df_mask['fit_error_u'] = fit_err_u_arr

        df_mask['chi2dof'] = chi2dof_arr
        df_mask['num_ps'] = num_ps_arr
        path_csv = Path('./fit_res/2048/PSCMBFGNOISE/csv')
        path_csv.mkdir(parents=True, exist_ok=True)
        df_mask.to_csv(path_csv / Path(f'{rlz_idx}.csv'), index=False)



# filter_df()
# calc_number()
# check_num_ps_overlap()
# get_disc_pix_ind()
# calc_cov()
# save_fit_res_to_csv()
save_pcfn_fit_res_to_csv()



