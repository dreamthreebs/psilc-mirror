import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pandas as pd
import time
import pickle
import os,sys
import logging
import ipdb
import gc

from pathlib import Path
from iminuit import Minuit
from iminuit.cost import LeastSquares

from fit_qu_base import FitPolPS

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.INFO)

freq = 270
lmax = 1999
nside = 2048
beam = 9

flux_idx = 0

def fit_rlz_2048():
    time0 = time.perf_counter()
    # m = np.load(f'../../fitdata/synthesis_data/2048/PSNOISE/{freq}/0.npy')

    logger.info(f'time for fitting = {time.perf_counter()-time0}')
    nstd = np.load(f'../../FGSim/NSTDNORTH/2048/{freq}.npy')
    nstd_q = nstd[1].copy()
    nstd_u = nstd[2].copy()
    df_mask = pd.read_csv(f'../mask/mask_csv/{freq}.csv')
    df_ps = pd.read_csv(f'../mask/ps_csv/{freq}.csv')

    rlz_num_ps_arr = np.zeros(100)
    rlz_chi2dof_arr = np.zeros(100)
    rlz_q_amp_arr = np.zeros(100)
    rlz_q_amp_err_arr = np.zeros(100)
    rlz_u_amp_arr = np.zeros(100)
    rlz_u_amp_err_arr = np.zeros(100)
    rlz_fit_err_q = np.zeros(100)
    rlz_fit_err_u = np.zeros(100)

    radius_factor = 1.5
    save_path = Path(f'fit_res/2048/PSCMBNOISE/{radius_factor}/idx_{flux_idx}')
    save_path.mkdir(exist_ok=True, parents=True)

    for rlz_idx in range(100):
        m = np.load(f'../../fitdata/synthesis_data/2048/PSCMBNOISE/{freq}/{rlz_idx}.npy').copy()
        m_q = m[1].copy()
        m_u = m[2].copy()

        obj = FitPolPS(m_q=m_q, m_u=m_u, freq=freq, nstd_q=nstd_q, nstd_u=nstd_u, flux_idx=flux_idx, df_mask=df_mask, df_ps=df_ps, lmax=lmax, nside=nside, radius_factor=radius_factor, beam=beam, epsilon=0.00001)
        num_ps, chi2dof, fit_q_amp, fit_q_amp_err, fit_u_amp, fit_u_amp_err, fit_error_q, fit_error_u = obj.fit_all(cov_mode='cmb+noise')

        del m
        del obj

        gc.collect()

        rlz_num_ps_arr[rlz_idx] = num_ps
        rlz_chi2dof_arr[rlz_idx] = chi2dof
        rlz_q_amp_arr[rlz_idx] = fit_q_amp
        rlz_q_amp_err_arr[rlz_idx] = fit_q_amp_err
        rlz_u_amp_arr[rlz_idx] = fit_u_amp
        rlz_u_amp_err_arr[rlz_idx] = fit_u_amp_err
        rlz_fit_err_q[rlz_idx] = fit_error_q
        rlz_fit_err_u[rlz_idx] = fit_error_u


    np.save(save_path / Path(f'num_ps.npy'), rlz_num_ps_arr)
    np.save(save_path / Path(f'chi2dof.npy'), rlz_chi2dof_arr)
    np.save(save_path / Path(f'q_amp.npy'), rlz_q_amp_arr)
    np.save(save_path / Path(f'q_amp_err.npy'), rlz_q_amp_err_arr)
    np.save(save_path / Path(f'u_amp.npy'), rlz_u_amp_arr)
    np.save(save_path / Path(f'u_amp_err.npy'), rlz_u_amp_err_arr)
    np.save(save_path / Path(f'fit_error_q.npy'), rlz_fit_err_q)
    np.save(save_path / Path(f'fit_error_u.npy'), rlz_fit_err_u)

fit_rlz_2048()




