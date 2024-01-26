import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pandas as pd
import time
import pickle
import os

from iminuit import Minuit
from iminuit.cost import LeastSquares
from numpy.polynomial.legendre import Legendre
from scipy.interpolate import CubicSpline

from fit_2048 import FitPointSource

def fit_PSNS_rlz_2048():

    df_ps = pd.read_csv('../partial_sky_ps/ps_in_mask/2048/40ps.csv')
    df_mask = pd.read_csv('../partial_sky_ps/ps_in_mask/2048/40mask.csv')
    lmax = 350
    nside = 2048
    beam = 63
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax)
    cl_cmb = np.load('../../src/cmbsim/cmbdata/cmbcl.npy')[:lmax+1,0]
    cl_cmb = cl_cmb * bl**2

    nstd = np.load('../../FGSim/NSTDNORTH/2048/40.npy')[0]
    flux_idx = 1
    lon = np.rad2deg(df_mask.at[flux_idx, 'lon'])
    lat = np.rad2deg(df_mask.at[flux_idx, 'lat'])
    iflux = df_mask.at[flux_idx, 'iflux']

    rlz_norm_beam_list = []
    rlz_norm_error_list = []
    rlz_chi2dof_list = []
    rlz_fit_error_list = []
    for rlz_idx in range(100):
        m = np.load(f'../../fitdata/synthesis_data/2048/PSNOISE/40/{rlz_idx}.npy')[0]

        obj = FitPointSource(m=m, nstd=nstd, flux_idx=flux_idx, df_mask=df_mask, df_ps=df_ps, cl_cmb=cl_cmb, lon=lon, lat=lat, iflux=iflux, lmax=lmax, nside=nside, radius_factor=1.5, beam=beam, epsilon=1e-5)

        # obj.see_true_map(m=m, lon=lon, lat=lat, nside=nside, beam=beam)

        obj.calc_covariance_matrix(mode='noise', cmb_cov_fold='../cov_r_1.5_2048/cov')
        # obj.calc_covariance_matrix(mode='cmb+noise', cmb_cov_fold='../cov_r_1.5_2048/cov')

        num_ps, chi2dof, norm_beam, norm_error, fit_lon, fit_lat, fit_error = obj.fit_all()
        rlz_norm_beam_list.append(norm_beam)
        rlz_norm_error_list.append(norm_error)
        rlz_chi2dof_list.append(chi2dof)
        rlz_fit_error_list.append(fit_error)

    breakpoint()

def fit_PSCMBNS_rlz():
    pass

fit_PSNS_rlz_2048()
