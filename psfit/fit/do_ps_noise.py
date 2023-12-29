import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pandas as pd
import time
import pickle

from iminuit import Minuit
from iminuit.cost import LeastSquares
from numpy.polynomial.legendre import Legendre
from scipy.interpolate import CubicSpline

# from fit_ps_base import FitPointSource
# from fitv1 import FitPointSource
from fitv2 import FitPointSource

def main():
    m = np.load('../../FGSim/PSNOISE/2048/40.npy')[0]
    nstd = np.load('../../FGSim/NSTDNORTH/2048/40.npy')[0]
    df_mask = pd.read_csv('../partial_sky_ps/ps_in_mask/mask40.csv')
    df_ps = pd.read_csv('../../test/ps_sort/sort_by_iflux/40.csv')

    lmax = 350
    nside = 2048
    beam = 63
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax)
    cl_cmb = np.load('../../src/cmbsim/cmbdata/cmbcl.npy')[:lmax+1,0]
    cl_cmb = cl_cmb * bl**2

    flux_idx = 0
    lon = np.rad2deg(df_mask.at[flux_idx, 'lon'])
    lat = np.rad2deg(df_mask.at[flux_idx, 'lat'])
    iflux = df_mask.at[flux_idx, 'iflux']

    obj = FitPointSource(m=m, nstd=nstd, flux_idx=flux_idx, df_mask=df_mask, df_ps=df_ps, cl_cmb=cl_cmb, lon=lon, lat=lat, iflux=iflux, lmax=lmax, nside=nside, radius_factor=1.0, beam=beam)

    obj.see_true_map(m=m, lon=lon, lat=lat, nside=nside, beam=beam)
    # obj.find_nearby_ps_lon_lat()
    # obj.find_first_second_nearby_ps_lon_lat()
    # obj.fit_ps_ns(mode='10params')
    norm_beam, fit_lon, fit_lat =  obj.fit_ps_ns()
    print(f'{norm_beam=}, {fit_lon=}, {fit_lat=}')

def main1():
    m = np.load('../../FGSim/PSNOISE/2048/40.npy')[0]
    nstd = np.load('../../FGSim/NSTDNORTH/2048/40.npy')[0]
    df_mask = pd.read_csv('../partial_sky_ps/ps_in_mask/mask40.csv')
    df_ps = pd.read_csv('../../test/ps_sort/sort_by_iflux/40.csv')

    lmax = 350
    nside = 2048
    beam = 63
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax)
    cl_cmb = np.load('../../src/cmbsim/cmbdata/cmbcl.npy')[:lmax+1,0]
    cl_cmb = cl_cmb * bl**2
    norm_beam_list = []
    fit_lon_list = []
    fit_lat_list = []

    for flux_idx in range(1,139):
        print(f'{flux_idx=}')
        lon = np.rad2deg(df_mask.at[flux_idx, 'lon'])
        lat = np.rad2deg(df_mask.at[flux_idx, 'lat'])
        iflux = df_mask.at[flux_idx, 'iflux']

        obj = FitPointSource(m=m, nstd=nstd, flux_idx=flux_idx, df_mask=df_mask, df_ps=df_ps, cl_cmb=cl_cmb, lon=lon, lat=lat, iflux=iflux, lmax=lmax, nside=nside, radius_factor=1.0, beam=beam)

        # obj.see_true_map(m=m, lon=lon, lat=lat, nside=nside, beam=beam)
        # obj.find_nearby_ps_lon_lat()
        # obj.find_first_second_nearby_ps_lon_lat()
        # obj.fit_ps_ns(mode='10params')
        norm_beam, fit_lon, fit_lat =  obj.fit_ps_ns()
        print(f'{norm_beam=}, {fit_lon=}, {fit_lat=}')
        norm_beam_list.append(norm_beam)
        fit_lon_list.append(fit_lon)
        fit_lat_list.append(fit_lat)


if __name__ == '__main__':
    main1()
    # main()

