import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from numpy.polynomial.legendre import Legendre
import time

lmax = 350
radius_fold = 0.8
print(f'{lmax=}')
print(f'{radius_fold=}')

def evaluate_interp_func(l, x, interp_funcs):
    for interp_func, x_range in interp_funcs[l]:
        if x_range[0] <= x <= x_range[1]:
            return interp_func(x)
    raise ValueError(f"x = {x} is out of the interpolation range for l = {l}")

def calc_C_theta_itp(x, lmax, cl, itp_funcs):
    sum_val = 0.0
    for l in range(lmax + 1):
        sum_val += (2 * l + 1) * cl[l] * evaluate_interp_func(l, x, interp_funcs=itp_funcs)
    return 1/(4*np.pi)*sum_val


# y = calc_C_theta(np.pi/2, lmax=2, cl=np.ones(lmax+1))
# print(f'{y=}')

beam = 63 # arcmin
sigma = np.deg2rad(beam)/60 / (np.sqrt(8*np.log(2)))
print(f'{sigma=}')

nside = 2048

m = np.load('../../FGSim/STRPSCMBFGNOISE/40.npy')[0]
# m = np.load('../../FGSim/STRPSFGNOISE/40.npy')[0]
# m = np.load('../../FGSim/STRPSCMBNOISE/40.npy')[0]
m = np.load('../../FGSim/PSNOISE/2048/40.npy')[0]
noise_nstd = np.load('../../FGSim/NSTDNORTH/2048/40.npy')[0]
cstd = np.ones(hp.nside2npix(nside)) *  75.2896

df = pd.read_csv('../ps_sort/sort_by_iflux/40.csv')
lon = df.at[44, 'lon']
lat = df.at[44, 'lat']
iflux = df.at[44, 'iflux']

center_pix = hp.ang2pix(nside=nside, theta=np.rad2deg(lon), phi=np.rad2deg(lat), lonlat=True)
# center_pix = 100000
print(f'{center_pix=}')
center_vec = hp.pix2vec(nside=nside, ipix=center_pix)
center_vec = np.array(center_vec).astype(np.float64)
print(f'{center_vec=}')

ipix_fit = hp.query_disc(nside=nside, vec=center_vec, radius=radius_fold * np.deg2rad(beam)/60)
print(f'{ipix_fit.shape=}')

n_cov = len(ipix_fit)
cov = np.zeros((n_cov, n_cov))
print(f'{cov.shape=}')

bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax)
cl = np.load('../../src/cmbsim/cmbdata/cmbcl.npy')[:lmax+1,0]
cl = cl * bl**2

time0 = time.time()
with open('../interpolate_cov/lgd_itp_funcs500.pkl', 'rb') as f:
    loaded_itp_funcs = pickle.load(f)
    for i in range(n_cov):
        print(f'{i=}')
        for j in range(i+1):
            ipix_i = ipix_fit[i]
            ipix_j = ipix_fit[j]
            vec_i = hp.pix2vec(nside=nside, ipix=ipix_i)
            vec_j = hp.pix2vec(nside=nside, ipix=ipix_j)
            cos_theta = np.array(vec_i) @ np.array(vec_j)
            cos_theta = np.where(cos_theta>1.0, 1.0, cos_theta)
            # print(f'{cos_theta=}')
            cov[i,j] = calc_C_theta_itp(x=cos_theta, lmax=lmax, cl=cl[0:lmax+1], itp_funcs=loaded_itp_funcs)
            if i!=j:
                cov[j,i] = cov[i,j]
timecov = time.time()-time0
print(f'{timecov=}')
print(f'{cov=}')
np.save(f'./cov_beam_itp_data/lmax{lmax}rf{radius_fold}', cov)











