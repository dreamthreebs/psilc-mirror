import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pandas as pd
import time
import pickle
import os
import sys
import ipdb
import logging

from pathlib import Path
from iminuit import Minuit
from iminuit.cost import LeastSquares
from numpy.polynomial.legendre import Legendre
from scipy.interpolate import CubicSpline
from memory_profiler import profile

from fit_base import FitPointSource

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.INFO)

def calc_cov(freq):
    m = np.load(f'../../fitdata/synthesis_data/2048/PSNOISE/{freq}/0.npy')[0]
    # m = np.load(f'../../fitdata/synthesis_data/2048/PSCMBNOISE/{freq}/1.npy')[0]
    # m = np.load(f'../../fitdata/synthesis_data/2048/CMBNOISE/{freq}/1.npy')[0]
    print(f'{sys.getrefcount(m)-1=}')
    nstd = np.load(f'../../FGSim/NSTDNORTH/2048/{freq}.npy')[0]
    df_mask = pd.read_csv(f'../mask/mask_csv/{freq}.csv')
    df_ps = pd.read_csv(f'../mask/ps_csv/{freq}.csv')
    lmax = 1999
    nside = 2048
    beam = 17
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax)
    cl_cmb = np.load('../../src/cmbsim/cmbdata/cmbcl.npy')[:lmax+1,0]

    cl_cmb = cl_cmb * bl**2

    for flux_idx in range(700):
        print(f'{flux_idx=}')
        lon = np.rad2deg(df_mask.at[flux_idx, 'lon'])
        lat = np.rad2deg(df_mask.at[flux_idx, 'lat'])
        iflux = df_mask.at[flux_idx, 'iflux']

        obj = FitPointSource(m=m, freq=freq, nstd=nstd, flux_idx=flux_idx, df_mask=df_mask, df_ps=df_ps, cl_cmb=cl_cmb, lon=lon, lat=lat, iflux=iflux, lmax=lmax, nside=nside, radius_factor=1.5, beam=beam, epsilon=1e-5)

        obj.calc_cmb_cov()
        obj.calc_definite_fixed_cmb_cov()
        obj.calc_covariance_matrix(mode='cmb+noise')

        obj.calc_covariance_matrix(mode='noise')

def check_all_def_fixed_cmb_cov():
    import glob
    file_list = glob.glob('./semi_def_cmb_cov_2048/r_1.5/*.npy')
    for idx, file in enumerate(file_list):
        logger.debug(f'{file=}')
        cov = np.load(file)
        cov  = cov + 0.00004 * np.eye(cov.shape[0])
        inv_cov = np.linalg.solve(cov, np.eye(cov.shape[0]))
        I = cov @ inv_cov
        logger.debug(f'{I=}')
        is_correct = np.allclose(I, np.eye(cov.shape[0]), atol=1e-3)
        logger.debug(f'{idx=}, {is_correct=}')

def check_overlap():
    m = np.load(f'../../fitdata/synthesis_data/2048/PSNOISE/{freq}/0.npy')[0]
    # m = np.load(f'../../fitdata/synthesis_data/2048/PSCMBNOISE/{freq}/1.npy')[0]
    # m = np.load(f'../../fitdata/synthesis_data/2048/CMBNOISE/{freq}/1.npy')[0]
    print(f'{sys.getrefcount(m)-1=}')
    nstd = np.load(f'../../FGSim/NSTDNORTH/2048/{freq}.npy')[0]
    df_mask = pd.read_csv(f'../mask/mask_csv/{freq}.csv')
    df_ps = pd.read_csv(f'../mask/ps_csv/{freq}.csv')
    lmax = 1999
    nside = 2048
    beam = 17
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax)
    cl_cmb = np.load('../../src/cmbsim/cmbdata/cmbcl.npy')[:lmax+1,0]

    cl_cmb = cl_cmb * bl**2
    overlap_list = []

    for flux_idx in range(700):
        print(f'{flux_idx=}')
        lon = np.rad2deg(df_mask.at[flux_idx, 'lon'])
        lat = np.rad2deg(df_mask.at[flux_idx, 'lat'])
        iflux = df_mask.at[flux_idx, 'iflux']

        obj = FitPointSource(m=m, freq=freq, nstd=nstd, flux_idx=flux_idx, df_mask=df_mask, df_ps=df_ps, cl_cmb=cl_cmb, lon=lon, lat=lat, iflux=iflux, lmax=lmax, nside=nside, radius_factor=1.5, beam=beam, epsilon=1e-5)

        flag = obj.fit_all(cov_mode='cmb+noise', mode='get_overlap')
        print(f'{flag=}')
        if flag == True:
            overlap_list.append(flux_idx)
    np.save('./overlap_arr.npy', np.array(overlap_list))

def see_overlap():
    overlap_arr = np.load('./overlap_arr.npy')
    print(f'{overlap_arr=}')


def save_fit_res_to_csv(freq):

    for rlz_idx in range(100):
        print(f'{rlz_idx=}')
        df= pd.read_csv(f'../mask/mask_csv/{freq}.csv', index_col=0)
        # print(f'{df=}')
        print(f'{df["iflux"]}')
        norm_true = FitPointSource.mJy_to_uKCMB(1, freq) * df["iflux"].to_numpy()

        fit_norm_arr = np.zeros(700)
        norm_error_arr = np.zeros(700)
        fit_error_arr = np.zeros(700)
        chi2dof_arr = np.zeros(700)
        num_ps_arr = np.zeros(700)


        for flux_idx in range(700):
            print(f'{flux_idx=}')
            fit_norm_arr[flux_idx] = np.load(f'./fit_res/2048/PSCMBNOISE/1.5/idx_{flux_idx}/norm_beam.npy')[rlz_idx]
            norm_error_arr[flux_idx] = np.load(f'./fit_res/2048/PSCMBNOISE/1.5/idx_{flux_idx}/norm_error.npy')[rlz_idx]
            fit_error_arr[flux_idx] = np.load(f'./fit_res/2048/PSCMBNOISE/1.5/idx_{flux_idx}/fit_error.npy')[rlz_idx]
            chi2dof_arr[flux_idx] = np.load(f'./fit_res/2048/PSCMBNOISE/1.5/idx_{flux_idx}/chi2dof.npy')[rlz_idx]
            num_ps_arr[flux_idx] = np.load(f'./fit_res/2048/PSCMBNOISE/1.5/idx_{flux_idx}/num_ps.npy')[rlz_idx]


        df['norm_true'] = norm_true
        df['fit_norm'] = fit_norm_arr
        df['norm_error'] = norm_error_arr
        df['fit_error'] = fit_error_arr
        df['chi2dof'] = chi2dof_arr
        df['num_ps'] = num_ps_arr

        path_csv = Path('./fit_res/2048/PSCMBNOISE/1.5/csv')
        path_csv.mkdir(parents=True, exist_ok=True)
        df.to_csv(path_csv / Path(f"{rlz_idx}.csv"), index=False)

freq = 155
# calc_cov(freq=freq)
# check_all_def_fixed_cmb_cov()
# check_overlap()
see_overlap()
# save_fit_res_to_csv(freq=freq)



