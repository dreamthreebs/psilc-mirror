import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import time
from scipy.interpolate import CubicSpline
from iminuit import Minuit
from iminuit.cost import LeastSquares

lmax = 350

beam = 63 # arcmin
sigma = np.deg2rad(beam)/60 / (np.sqrt(8*np.log(2)))
print(f'{sigma=}')

nside = 2048

# m = np.load('../../FGSim/STRPSCMBFGNOISE/40.npy')[0]
# m = np.load('../../FGSim/STRPSFGNOISE/40.npy')[0]
m = np.load('../../FGSim/STRPSCMBNOISE/40.npy')[0]
# m = np.load('../../FGSim/PSNOISE/2048/40.npy')[0]
nstd = np.load('../../FGSim/NSTDNORTH/2048/40.npy')[0]

df = pd.read_csv('../partial_sky_ps/ps_in_mask/mask40.csv')
lon = df.at[1, 'lon']
lat = df.at[1, 'lat']
iflux = df.at[1, 'iflux']

print(f'{iflux=}')

def see_true_map():
   
    
    # hp.mollview(m, norm='hist');plt.show()
    
    hp.gnomview(m, rot=[np.rad2deg(lon), np.rad2deg(lat), 0])
    plt.show()
    
    vec = hp.ang2vec(theta=np.rad2deg(lon), phi=np.rad2deg(lat), lonlat=True)
    
    ipix_disc = hp.query_disc(nside=nside, vec=vec, radius=np.deg2rad(beam)/60)
    
    mask = np.ones(hp.nside2npix(nside))
    mask[ipix_disc] = 0
    
    hp.gnomview(mask, rot=[np.rad2deg(lon), np.rad2deg(lat), 0])
    plt.show()

# see_true_map()

center_pix = hp.ang2pix(nside=nside, theta=np.rad2deg(lon), phi=np.rad2deg(lat), lonlat=True)
# center_pix = 100000
print(f'{center_pix=}')
center_vec = hp.pix2vec(nside=nside, ipix=center_pix)
center_vec = np.array(center_vec).astype(np.float64)
print(f'{center_vec=}')

ipix_fit = hp.query_disc(nside=nside, vec=center_vec, radius=1.0 * np.deg2rad(beam)/60)
print(f'{ipix_fit.shape=}')

lon_fit = df.at[1, 'lon']
lat_fit = df.at[1, 'lat']

vec_around = hp.pix2vec(nside=nside, ipix=ipix_fit.astype(int))
theta = np.arccos(center_vec @ np.array(vec_around))
theta = np.nan_to_num(theta)

def evaluate_interp_func(l, x, interp_funcs):
    for interp_func, x_range in interp_funcs[l]:
        if x_range[0] <= x <= x_range[1]:
            return interp_func(x)
    raise ValueError(f"x = {x} is out of the interpolation range for l = {l}")

def calc_C_theta_itp1(x, lmax, cl, itp_funcs):
    Pl = np.zeros(lmax+1)
    for l in range(lmax+1):
        Pl[l] = evaluate_interp_func(l, x, interp_funcs=itp_funcs)

    # Pl = np.array([evaluate_interp_func(l, x, interp_funcs=itp_funcs) for l in np.arange(lmax+1)])
    ell = np.arange(lmax+1)
    sum_val = 1 / (4 * np.pi) * np.sum((2 * ell + 1) * cl * Pl)
    return sum_val

with open('../../test/interpolate_cov/lgd_itp_funcs350.pkl', 'rb') as f:
    loaded_itp_funcs = pickle.load(f)

n_cov = len(ipix_fit)
cov = np.zeros((n_cov,n_cov))
print(f'{cov.shape=}')
bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax)
cl = np.load('../../src/cmbsim/cmbdata/cmbcl.npy')[:lmax+1,0]
cl = cl * bl**2

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

# cov = np.zeros((len(ipix_fit), len(ipix_fit)))
nstd2 = (nstd**2)[ipix_fit]
for i in range(len(ipix_fit)):
    cov[i,i] = cov[i,i] + nstd2[i]
print(f'{cov.shape=}')
inv_cov = np.linalg.inv(cov)

def model1(theta, norm_beam, const):
    return norm_beam / (2*np.pi*sigma**2) * np.exp(- (theta)**2 / (2 * sigma**2)) + const


def lsq(norm_beam, const):
    y_model = model1(theta, norm_beam, const)
    print(f'{y_model.shape=}')

    y_data = m[ipix_fit]
    # print(f'{y_data.shape=}')

    y_diff = y_data - y_model

    z = (y_diff) @ inv_cov @ (y_diff)
    return z

obj_minuit = Minuit(lsq, norm_beam=0.2476,  const=0)
obj_minuit.limits = [(0,10),(-1e4,1e4)]
# print(obj_minuit.scan(ncall=1000))
# obj_minuit.errors = (0.1, 0.2)
print(obj_minuit.migrad())
print(obj_minuit.hesse())
ndof = len(ipix_fit)
str_chi2 = f"𝜒²/ndof = {obj_minuit.fval:.2f} / {ndof} = {obj_minuit.fval/ndof}"
print(str_chi2)


# fit_res = model1(theta, 0.2476 , -10)

# new_m = np.zeros(hp.nside2npix(nside))
# new_m[ipix_fit] = fit_res
# true_m = np.zeros(hp.nside2npix(nside))
# true_m[ipix_fit] = m[ipix_fit]
# hp.gnomview(new_m, rot=[np.rad2deg(lon), np.rad2deg(lat), 0])
# hp.gnomview(true_m, rot=[np.rad2deg(lon), np.rad2deg(lat), 0])
# hp.gnomview(true_m-new_m, rot=[np.rad2deg(lon), np.rad2deg(lat), 0], title='residual')
# plt.show()



# plt.plot(theta, fit_res, label='hand')
# plt.plot(theta, m[ipix_fit], label='true')
# plt.legend()
# plt.show()





