import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pandas as pd
import time
import pickle
from numpy.polynomial.legendre import Legendre
from scipy.interpolate import CubicSpline

lmax = 350
radius_fold = 1.0
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

def calc_C_theta_itp1(x, lmax, cl, itp_funcs):

    Pl = np.zeros(lmax+1)
    for l in range(lmax+1):
        Pl[l] = evaluate_interp_func(l, x, interp_funcs=itp_funcs)

    # Pl = np.array([evaluate_interp_func(l, x, interp_funcs=itp_funcs) for l in np.arange(lmax+1)])
    ell = np.arange(lmax+1)
    sum_val = 1 / (4 * np.pi) * np.sum((2 * ell + 1) * cl * Pl)
    return sum_val



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


with open('../interpolate_cov/lgd_itp_funcs350.pkl', 'rb') as f:
    loaded_itp_funcs = pickle.load(f)

# cos_theta = 0.99998
cos_theta_list = np.linspace(0.99, 1, 2000)
C_theta_list = []
time0 = time.time()
for cos_theta in cos_theta_list:
    C_theta = calc_C_theta_itp1(x=cos_theta, lmax=lmax, cl=cl[0:lmax+1], itp_funcs=loaded_itp_funcs)
    C_theta_list.append(C_theta)
print(f'{C_theta_list=}')
timecov = time.time()-time0
print(f'{timecov=}')

cs = CubicSpline(cos_theta_list, C_theta_list)

theta_cache = {}
time0 = time.time()
with open('../interpolate_cov/lgd_itp_funcs350.pkl', 'rb') as f:
    loaded_itp_funcs = pickle.load(f)
    for i in range(n_cov):
        print(f'{i=}')
        for j in range(i+1):
            if i == j:
                cov[i, i] = 1 / (4 * np.pi) * np.sum((2 * np.arange(lmax + 1) + 1) * cl[:lmax+1])
            else:
                ipix_i = ipix_fit[i]
                ipix_j = ipix_fit[j]
                vec_i = hp.pix2vec(nside=nside, ipix=ipix_i)
                vec_j = hp.pix2vec(nside=nside, ipix=ipix_j)
                cos_theta = np.dot(vec_i, vec_j)  # Assuming this results in a single value
                cos_theta = min(1.0, max(cos_theta, -1.0))  # Ensuring it's within [-1, 1]
    
                # Use cos_theta as a key in the dictionary
                if cos_theta not in theta_cache:
                    cov_ij = cs(cos_theta)
                    theta_cache[cos_theta] = cov_ij
                else:
                    cov_ij = theta_cache[cos_theta]
    
                cov[i, j] = cov_ij
                cov[j, i] = cov_ij

timecov = time.time()-time0
print(f'{timecov=}')
print(f'{cov=}')

np.save(f'./cov_beam_optimized_data/lmax{lmax}rf{radius_fold}2', cov)














