import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pandas as pd
import time
import pickle
import os

from pathlib import Path
from iminuit import Minuit
from iminuit.cost import LeastSquares
from numpy.polynomial.legendre import Legendre
from scipy.interpolate import CubicSpline

from fit_2048 import FitPointSource

def main():
    # m = np.load('../../fitdata/synthesis_data/2048/PSNOISE/40/1.npy')[0]
    m = np.load('../../fitdata/synthesis_data/2048/PSCMBFGNOISE/40/2.npy')[0]
    nstd = np.load('../../FGSim/NSTDNORTH/2048/40.npy')[0]
    df_mask = pd.read_csv('../partial_sky_ps/ps_in_mask/2048/40mask.csv')
    df_ps = pd.read_csv('../partial_sky_ps/ps_in_mask/2048/40ps.csv')
    lmax = 350
    nside = 2048
    beam = 63
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax)
    cl_cmb = np.load('../../src/cmbsim/cmbdata/cmbcl.npy')[:lmax+1,0]
    cl_cmb = cl_cmb * bl**2

    for flux_idx in range(136):
        print(f'{flux_idx=}')
        lon = np.rad2deg(df_mask.at[flux_idx, 'lon'])
        lat = np.rad2deg(df_mask.at[flux_idx, 'lat'])
        iflux = df_mask.at[flux_idx, 'iflux']

        obj = FitPointSource(m=m, nstd=nstd, flux_idx=flux_idx, df_mask=df_mask, df_ps=df_ps, cl_cmb=cl_cmb, lon=lon, lat=lat, iflux=iflux, lmax=lmax, nside=nside, radius_factor=2.0, beam=beam, epsilon=1e-5)

        # obj.see_true_map(m=m, lon=lon, lat=lat, nside=nside, beam=beam)

        obj.calc_C_theta()
        obj.calc_covariance_matrix(mode='cmb+noise')

        # obj.fit_all(cov_mode='cmb+noise')
def main1():
    # m = np.load('../../fitdata/synthesis_data/2048/PSNOISE/40/1.npy')[0]
    m = np.load('../../fitdata/synthesis_data/2048/PSCMBFGNOISE/40/2.npy')[0]
    nstd = np.load('../../FGSim/NSTDNORTH/2048/40.npy')[0]
    df_mask = pd.read_csv('../partial_sky_ps/ps_in_mask/2048/40mask.csv')
    df_ps = pd.read_csv('../partial_sky_ps/ps_in_mask/2048/40ps.csv')
    lmax = 350
    nside = 2048
    beam = 63
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax)
    cl_cmb = np.load('../../src/cmbsim/cmbdata/cmbcl.npy')[:lmax+1,0]
    cl_cmb = cl_cmb * bl**2

    for flux_idx in range(136):
        print(f'{flux_idx=}')
        lon = np.rad2deg(df_mask.at[flux_idx, 'lon'])
        lat = np.rad2deg(df_mask.at[flux_idx, 'lat'])
        iflux = df_mask.at[flux_idx, 'iflux']

        obj = FitPointSource(m=m, nstd=nstd, flux_idx=flux_idx, df_mask=df_mask, df_ps=df_ps, cl_cmb=cl_cmb, lon=lon, lat=lat, iflux=iflux, lmax=lmax, nside=nside, radius_factor=2.0, beam=beam, epsilon=1e-5)

        # obj.see_true_map(m=m, lon=lon, lat=lat, nside=nside, beam=beam)

        obj.calc_C_theta()
        obj.calc_covariance_matrix(mode='noise')

        # obj.fit_all(cov_mode='cmb+noise')


main()
