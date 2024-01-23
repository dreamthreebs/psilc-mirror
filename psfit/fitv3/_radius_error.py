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

from fit_base_256 import FitPointSource

if __name__ == '__main__':
    # m = np.load('../../FGSim/FITDATA/256/PSNOISE/40.npy')[0]
    m = np.load('../../FGSim/TEST0118/256/PSCMB/40.npy')[0]
    nstd = np.load('../../FGSim/NSTDNORTH/256/40.npy')[0]
    df_mask = pd.read_csv('../partial_sky_ps/ps_in_mask/2048/40mask.csv')
    flux_idx = 1
    lon = np.rad2deg(df_mask.at[flux_idx, 'lon'])
    lat = np.rad2deg(df_mask.at[flux_idx, 'lat'])
    iflux = df_mask.at[flux_idx, 'iflux']

    df_ps = pd.read_csv('../partial_sky_ps/ps_in_mask/2048/40ps.csv')
    
    lmax = 350
    nside = 256
    beam = 63
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

    
    norm_error_list = []
    chi2dof_list = []
    fit_error_list = []
    radius_factor_list = np.linspace(0.5, 3.2, 20)
    for radius_factor in radius_factor_list:
        obj = FitPointSource(m=m, nstd=nstd, flux_idx=flux_idx, df_mask=df_mask, df_ps=df_ps, cl_cmb=cl_cmb, lon=lon, lat=lat, iflux=iflux, lmax=lmax, nside=nside, radius_factor=radius_factor, beam=beam, epsilon=1e-5)

        # obj.see_true_map(m=m, lon=lon, lat=lat, nside=nside, beam=beam)

        # obj.calc_covariance_matrix(mode='noise', cmb_cov_fold='../cmb_cov_calc/cov')

        obj.calc_C_theta(save_path='./cov_256')
        obj.calc_covariance_matrix(mode='cmb+noise', cmb_cov_fold='./cov_256')
        obj.calc_covariance_matrix(mode='cmb', cmb_cov_fold='./cov_256')

        _, chi2dof, _, norm_error, _, _, fit_error = obj.fit_all()
        norm_error_list.append(norm_error)
        chi2dof_list.append(chi2dof)
        fit_error_list.append(fit_error)

    plt.figure(1)
    plt.plot(radius_factor_list, norm_error_list)
    plt.xlabel('fit radius (unit in beam size)')
    plt.ylabel('norm_error')
    plt.figure(2)
    plt.plot(radius_factor_list, chi2dof_list)
    plt.xlabel('fit radius (unit in beam size)')
    plt.ylabel('chi2 / ndof')
    plt.figure(3)
    plt.plot(radius_factor_list, fit_error_list)
    plt.xlabel('fit radius (unit in beam size)')
    plt.ylabel('fit error')

    plt.show()



