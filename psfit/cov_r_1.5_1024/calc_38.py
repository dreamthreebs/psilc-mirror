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

from fit_1024 import FitPointSource

def main():
    m = np.load('../../fitdata/synthesis_data/1024/PSNOISE/40/0.npy')[0]
    nstd = np.load('../../FGSim/NSTDNORTH/1024/40.npy')[0]
    df_mask = pd.read_csv('../partial_sky_ps/ps_in_mask/2048/40mask.csv')
    flux_idx = 38
    lon = np.rad2deg(df_mask.at[flux_idx, 'lon'])
    lat = np.rad2deg(df_mask.at[flux_idx, 'lat'])
    iflux = df_mask.at[flux_idx, 'iflux']

    df_ps = pd.read_csv('../partial_sky_ps/ps_in_mask/2048/40ps.csv')
    
    lmax = 350
    nside = 1024
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

    obj = FitPointSource(m=m, nstd=nstd, flux_idx=flux_idx, df_mask=df_mask, df_ps=df_ps, cl_cmb=cl_cmb, lon=lon, lat=lat, iflux=iflux, lmax=lmax, nside=nside, radius_factor=1.5, beam=beam)

    # obj.see_true_map(m=m, lon=lon, lat=lat, nside=nside, beam=beam)

    # obj.calc_C_theta_itp_func('../../test/interpolate_cov/lgd_itp_funcs350.pkl')
    if not os.path.exists('./cov'):
        os.makedirs('./cov')

    obj.calc_C_theta(save_path='./cov')

    # obj.calc_covariance_matrix(mode='cmb+noise', cmb_cov_fold='./covlmin/180')
    # obj.fit_all()

main()

