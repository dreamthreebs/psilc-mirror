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

from fit_test_cmb import FitPointSource

def main():

    rlz_list = []
    for idx_rlz in range(10):
        m = np.load(f'../../inpaintingdata/CMBREALIZATION/40GHz/{idx_rlz}.npy')[0]
        nstd = np.load('../../FGSim/NSTDNORTH/2048/40.npy')[0]
        df_mask = pd.read_csv('../partial_sky_ps/ps_in_mask/mask40.csv')
        flux_idx = 1
        lon = np.rad2deg(df_mask.at[flux_idx, 'lon'])
        lat = np.rad2deg(df_mask.at[flux_idx, 'lat'])
        iflux = df_mask.at[flux_idx, 'iflux']

        df_ps = pd.read_csv('../../test/ps_sort/sort_by_iflux/40.csv')
        
        lmax = 350
        nside = 2048
        beam = 63
        bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax)
        # m = np.load('../../inpaintingdata/CMB8/40.npy')[0]
        # cl1 = hp.anafast(m, lmax=lmax)
        cl_cmb = np.load('../../src/cmbsim/cmbdata/cmbcl.npy')[:lmax+1,0]
        l = np.arange(lmax+1)

        # plt.plot(l*(l+1)*cl_cmb/(2*np.pi))
        cl_cmb = cl_cmb * bl**2

        # plt.plot(l*(l+1)*cl_cmb/(2*np.pi))
        # plt.plot(l*(l+1)*cl1/(2*np.pi), label='cl1')
        # plt.show()

        obj = FitPointSource(m=m, nstd=nstd, flux_idx=flux_idx, df_mask=df_mask, df_ps=df_ps, cl_cmb=cl_cmb, lon=lon, lat=lat, iflux=iflux, lmax=lmax, nside=nside, radius_factor=1.0, beam=beam)

        # obj.see_true_map(m=m, lon=lon, lat=lat, nside=nside, beam=beam)

        # obj.calc_C_theta_itp_func('../../test/interpolate_cov/lgd_itp_funcs350.pkl')
        # obj.calc_C_theta()
        # obj.calc_covariance_matrix(mode='cmb', cmb_cov_fold='./cov')

        chi2dof_list = []
        for j in range(155, 180):
            obj.calc_covariance_matrix(mode='cmb', cmb_cov_fold=f'./covlmin/{j}')
            chi2dof = obj.fit_all()
            chi2dof_list.append(chi2dof)
        chi2dof_arr = np.array(chi2dof_list)
        rlz_list.append(chi2dof_arr)

    rlz_arr = np.array(rlz_list)
    return rlz_arr

rlz_arr = main()
np.save('rlz_arr.npy', rlz_arr)



