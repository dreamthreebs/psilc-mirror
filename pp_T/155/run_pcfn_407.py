import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pandas as pd
import time
import pickle
import os,sys
import gc
import logging

from iminuit import Minuit
from iminuit.cost import LeastSquares
from numpy.polynomial.legendre import Legendre
from scipy.interpolate import CubicSpline
from pathlib import Path
from fit_base import FitPointSource

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.INFO)

def fit_rlz_2048():

    freq = 155
    beam = 17
    df_ps = pd.read_csv(f'../mask/ps_csv/{freq}.csv')
    df_mask = pd.read_csv(f'../mask/mask_csv/{freq}.csv')
    lmax = 1999
    nside = 2048
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax)
    cl_cmb = np.load('../../src/cmbsim/cmbdata/cmbcl.npy')[:lmax+1,0]
    cl_cmb = cl_cmb * bl**2
    nstd = np.load(f'../../FGSim/NSTDNORTH/2048/{freq}.npy')[0]
    flux_idx = 407
    lon = np.rad2deg(df_mask.at[flux_idx, 'lon'])
    lat = np.rad2deg(df_mask.at[flux_idx, 'lat'])
    iflux = df_mask.at[flux_idx, 'iflux']

    rlz_norm_beam_arr = np.zeros(100)
    rlz_norm_error_arr = np.zeros(100)
    rlz_chi2dof_arr = np.zeros(100)
    rlz_fit_error_arr = np.zeros(100)
    rlz_num_ps_arr = np.zeros(100)
    radius_factor = 1.5
    save_path = Path(f'fit_res/2048/PSCMBFGNOISE/{radius_factor}/idx_{flux_idx}')
    # save_path = Path(f'fit_res/2048/PSCMBNOISE/check_GOF/idx_{flux_idx}')
    save_path.mkdir(parents=True, exist_ok=True)

    for rlz_idx in range(100):
        print(f'{rlz_idx=}')
        m = np.load(f'../../fitdata/synthesis_data/2048/PSCMBFGNOISE/{freq}/{rlz_idx}.npy')[0]
        print(f'before new object{sys.getrefcount(m)-1=}')

        obj = FitPointSource(m=m, nstd=nstd, freq=freq, flux_idx=flux_idx, df_mask=df_mask, df_ps=df_ps, cl_cmb=cl_cmb, lon=lon, lat=lat, iflux=iflux, lmax=lmax, nside=nside, radius_factor=radius_factor, beam=beam, epsilon=1e-5)

        print(f'before fit_all{sys.getrefcount(m)-1=}')
        print(f'before fit_all{sys.getrefcount(obj)-1=}')
        # obj.see_true_map(m=m, lon=lon, lat=lat, nside=nside, beam=beam)

        # obj.calc_covariance_matrix(mode='noise', cmb_cov_fold='../cov_r_1.5_2048/cov')
        # obj.calc_covariance_matrix(mode='cmb+noise', cmb_cov_fold='../cov_r_1.5_2048/cov')

        num_ps, chi2dof, norm_beam, norm_error, fit_error = obj.fit_all(cov_mode='cmb+noise')
        # num_ps, chi2dof, norm_beam, norm_error, fit_error = obj.fit_all(cov_mode='noise')

        print(f'after fit_all{sys.getrefcount(m)-1=}')
        print(f'after fit_all{sys.getrefcount(obj)-1=}')
        del m
        del obj

        gc.collect()
        rlz_norm_beam_arr[rlz_idx] = norm_beam
        rlz_norm_error_arr[rlz_idx] = norm_error
        rlz_chi2dof_arr[rlz_idx] = chi2dof
        rlz_fit_error_arr[rlz_idx] = fit_error
        rlz_num_ps_arr[rlz_idx] = num_ps

    np.save(save_path / f'norm_beam.npy', rlz_norm_beam_arr)
    np.save(save_path / f'norm_error.npy', rlz_norm_error_arr)
    np.save(save_path / f'chi2dof.npy', rlz_chi2dof_arr)
    np.save(save_path / f'fit_error.npy', rlz_fit_error_arr)
    np.save(save_path / f'num_ps.npy', rlz_num_ps_arr)

# gc.set_debug(gc.DEBUG_LEAK)
fit_rlz_2048()
# fit_PSCMBNS_rlz_normalize_noise()

