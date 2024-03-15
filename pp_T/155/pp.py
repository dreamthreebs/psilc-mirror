import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pandas as pd
import time
import pickle
import os
import sys
import ipdb

from pathlib import Path
from iminuit import Minuit
from iminuit.cost import LeastSquares
from numpy.polynomial.legendre import Legendre
from scipy.interpolate import CubicSpline
from memory_profiler import profile

from fit_2048 import FitPointSource

def calc_cov():
    m = np.load('../../fitdata/synthesis_data/2048/PSNOISE/155/0.npy')[0]
    # m = np.load('../../fitdata/synthesis_data/2048/PSCMBNOISE/40/1.npy')[0]
    # m = np.load('../../fitdata/synthesis_data/2048/CMBNOISE/40/1.npy')[0]
    print(f'{sys.getrefcount(m)-1=}')
    nstd = np.load('../../FGSim/NSTDNORTH/2048/155.npy')[0]
    # df_mask = pd.read_csv('../../psfit/partial_sky_ps/ps_in_mask/2048/40mask.csv')
    df_mask = pd.read_csv('../mask/mask_csv/155.csv')
    # df_ps = pd.read_csv('../../psfit/partial_sky_ps/ps_in_mask/2048/40ps.csv')
    df_ps = pd.read_csv('../mask/ps_csv/155.csv')
    freq = 155
    lmax = 1999
    nside = 2048
    beam = 17
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

    for flux_idx in range(700):
        print(f'{flux_idx=}')
        lon = np.rad2deg(df_mask.at[flux_idx, 'lon'])
        lat = np.rad2deg(df_mask.at[flux_idx, 'lat'])
        iflux = df_mask.at[flux_idx, 'iflux']

        obj = FitPointSource(m=m, freq=freq, nstd=nstd, flux_idx=flux_idx, df_mask=df_mask, df_ps=df_ps, cl_cmb=cl_cmb, lon=lon, lat=lat, iflux=iflux, lmax=lmax, nside=nside, radius_factor=1.5, beam=beam, epsilon=1e-5)

        # obj.calc_C_theta()
        # obj.calc_covariance_matrix(mode='cmb+noise')
        obj.calc_covariance_matrix(mode='noise')

calc_cov()

