import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pandas as pd
import time
import pickle
import os
import sys

# from pathlib import Path
# from iminuit import Minuit
# from iminuit.cost import LeastSquares
# from numpy.polynomial.legendre import Legendre
# from scipy.interpolate import CubicSpline

from fit_2048 import FitPointSource

def main():
    # m = np.load('../../fitdata/synthesis_data/2048/PSNOISE/40/1.npy')[0]
    m = np.load('../../fitdata/synthesis_data/2048/PSNOISE/155/0.npy')[0]
    print(f'{sys.getrefcount(m)-1=}')
    nstd = np.load('../../FGSim/NSTDNORTH/2048/155.npy')[0]
    df_mask = pd.read_csv('../mask/mask_csv/155.csv')
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

    num_ps_list = []
    num_near_ps_list = []
    ang_near_list = []
    for flux_idx in range(700):
        print(f'{flux_idx=}')
        lon = np.rad2deg(df_mask.at[flux_idx, 'lon'])
        lat = np.rad2deg(df_mask.at[flux_idx, 'lat'])
        iflux = df_mask.at[flux_idx, 'iflux']

    
        print(f'{sys.getrefcount(m)-1=}')
        obj = FitPointSource(m=m, freq=155, nstd=nstd, flux_idx=flux_idx, df_mask=df_mask, df_ps=df_ps, cl_cmb=cl_cmb, lon=lon, lat=lat, iflux=iflux, lmax=lmax, nside=nside, radius_factor=1.5, beam=beam, epsilon=1e-5)

        print(f'{sys.getrefcount(m)-1=}')
        # obj.see_true_map(m=m, lon=lon, lat=lat, nside=nside, beam=beam)

        # obj.calc_covariance_matrix(mode='noise', cmb_cov_fold='../cmb_cov_calc/cov')

        # obj.calc_C_theta_itp_func(lgd_itp_func_pos='../../test/interpolate_cov/lgd_itp_funcs350.pkl')
        # obj.calc_C_theta(save_path='./cov_r_2.0/2048')
        # obj.calc_precise_C_theta()

        # obj.calc_C_theta()
        # obj.calc_covariance_matrix(mode='cmb+noise')

        # time0 = time.perf_counter()
        num_ps, num_near_ps, ang_near = obj.fit_all(cov_mode='noise', mode='get_num_ps')
        # print(f'{time.perf_counter()-time0}')
        num_ps_list.append(num_ps)
        num_near_ps_list.append(num_near_ps)
        ang_near_list.append(ang_near)


    np.save('num_ps.npy', np.array(num_ps_list))
    np.save('num_near_ps.npy', np.array(num_near_ps_list))
    with open('./ang_near.pkl', 'wb') as f:
        pickle.dump(ang_near_list, f)

# main()

numps = np.load('./num_ps.npy')
num_near_ps = np.load('./num_near_ps.npy')
print(f'{numps=}')
print(f'{num_near_ps=}')
with open('./ang_near.pkl', 'rb') as f:
    ang_near_list = pickle.load(f)

print(f'{ang_near_list=}')

# Find the length of the longest array
max_len = max(len(arr) for arr in ang_near_list)

# Create a DataFrame with the appropriate shape, filling with NaN for missing values
df = pd.DataFrame([list(arr) + [np.nan]*(max_len - len(arr)) for arr in ang_near_list])

# Save the DataFrame to a CSV file
df.to_csv('ang_near_list.csv', index=False, header=False)

