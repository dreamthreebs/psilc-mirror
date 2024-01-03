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

from fitcmbfg import FitPointSource

if __name__ == "__main__":
    m = np.load('../../FGSim/STRPSCMBFGNOISE/40.npy')[0]
    nstd = np.load('../../FGSim/NSTDNORTH/2048/40.npy')[0]
    df_mask = pd.read_csv('../partial_sky_ps/ps_in_mask/mask40.csv')
    for i in range(70,139):
        flux_idx = i
        lon = np.rad2deg(df_mask.at[flux_idx, 'lon'])
        lat = np.rad2deg(df_mask.at[flux_idx, 'lat'])
        iflux = df_mask.at[flux_idx, 'iflux']

        df_ps = pd.read_csv('../../test/ps_sort/sort_by_iflux/40.csv')
        
        lmax = 350
        nside = 2048
        beam = 63
        bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax)
        cl_cmb = np.load('../../src/cmbsim/cmbdata/cmbcl.npy')[:lmax+1,0]
        cl_cmb = cl_cmb * bl**2

        obj = FitPointSource(m=m, nstd=nstd, flux_idx=flux_idx, df_mask=df_mask, df_ps=df_ps, cl_cmb=cl_cmb, lon=lon, lat=lat, iflux=iflux, lmax=lmax, nside=nside, radius_factor=1.0, beam=beam)

        # obj.see_true_map(m=m, lon=lon, lat=lat, nside=nside, beam=beam)
        # obj.fit_ps()
        obj.calc_C_theta('../../test/interpolate_cov/lgd_itp_funcs350.pkl')
        # obj.calc_covariance_matrix()
        # obj.fit_ps()

