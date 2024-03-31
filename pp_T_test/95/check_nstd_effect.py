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
from numpy.polynomial.legendre import Legendre
from scipy.interpolate import CubicSpline
from memory_profiler import profile

# logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s -%(name)s - %(message)s')
logging.basicConfig(level=logging.WARNING)
# logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.INFO)

from fit_2 import FitPointSource

def main():
    freq = 95
    time0 = time.perf_counter()
    # m = np.load(f'../../fitdata/synthesis_data/2048/PSNOISE/{freq}/800.npy')[0]
    # m = np.load(f'../../fitdata/synthesis_data/2048/PSCMBNOISE/{freq}/0.npy')[0]
    # m = np.load(f'../../fitdata/synthesis_data/2048/CMBNOISE/{freq}/0.npy')[0]
    m = np.load(f'../../fitdata/synthesis_data/2048/CMBNOISE/155_to_95/0.npy')
    logger.debug(f'{sys.getrefcount(m)-1=}')


    logger.info(f'time for fitting = {time.perf_counter()-time0}')
    # nstd = np.load(f'../../FGSim/NSTDNORTH/2048/{freq}.npy')[0]
    # print(f'{nstd[0]=}')
    df_mask = pd.read_csv(f'../mask/mask_csv/{freq}.csv')
    df_ps = pd.read_csv(f'../mask/ps_csv/{freq}.csv')
    lmax = 1999
    nside = 2048
    beam = 30
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax)
    # m = np.load('../../inpaintingdata/CMB8/40.npy')[0]
    # cl1 = hp.anafast(m, lmax=lmax)
    cl_cmb = np.load('../../src/cmbsim/cmbdata/cmbcl.npy')[:lmax+1,0]
    # l = np.arange(lmax+1)

    # plt.plot(l*(l+1)*cl_cmb/(2*np.pi))
    cl_cmb = cl_cmb * bl**2

    # plt.plot(l*(l+1)*cl_cmb/(2*np.pi))
    # plt.plot(l*(l+1)*cl1/(2*np.pi), label='cl1')
    # plt.show()

    flux_idx = 1
    lon = np.rad2deg(df_mask.at[flux_idx, 'lon'])
    lat = np.rad2deg(df_mask.at[flux_idx, 'lat'])
    iflux = df_mask.at[flux_idx, 'iflux']

    fit_norm_list = []
    chi2dof_list = []
    for nstd_fold in np.linspace(1,100,20):
        nstd = np.load(f'../../FGSim/NSTDNORTH/2048/{freq}.npy')[0] * nstd_fold

        obj = FitPointSource(m=m, freq=freq, nstd=nstd, flux_idx=flux_idx, df_mask=df_mask, df_ps=df_ps, cl_cmb=cl_cmb, lon=lon, lat=lat, iflux=iflux, lmax=lmax, nside=nside, radius_factor=1.5, beam=beam, epsilon=1e-5)

        # obj.see_true_map(m=m, lon=lon, lat=lat, nside=nside, beam=beam)

        # obj.calc_covariance_matrix(mode='noise', cmb_cov_fold='../cmb_cov_calc/cov')

        # obj.calc_C_theta_itp_func()
        # obj.calc_C_theta(save_path='./cov_r_2.0/2048')
        # obj.calc_precise_C_theta()

        # obj.calc_C_theta()
        obj.calc_covariance_matrix(mode='noise')
        # obj.calc_covariance_matrix(mode='cmb+noise')

        # obj.fit_all(cov_mode='cmb+noise')
        num_ps, chi2dof, fit_norm, norm_error, fit_error = obj.fit_all(cov_mode='noise')
        fit_norm_list.append(fit_norm)
        chi2dof_list.append(chi2dof)

    plt.figure(1)
    plt.plot(np.linspace(1,100,20), fit_norm_list, label='fit_norm')
    plt.xlabel('normalization factor on noise')
    plt.ylabel('point source amplitude')
    plt.figure(2)
    plt.semilogy(np.linspace(1,100,20), chi2dof_list, label='chi2dof')
    plt.xlabel('normalization factor on noise')
    plt.ylabel('chi2dof')
    # plt.legend()
    plt.show()


main()
