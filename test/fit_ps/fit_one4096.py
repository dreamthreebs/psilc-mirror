import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares

beam = 63 # arcmin
sigma = np.deg2rad(beam)/60 / (np.sqrt(8*np.log(2)))

nside = 4096

m = np.load('../../FGSim/PSNOISE/4096/40psnoise.npy')[0]
noise_nstd = np.load('../../FGSim/PSNOISE/4096/40nstd.npy')[0]

df = pd.read_csv('../ps_sort/sort_by_iflux/40.csv')
lon = df.at[5, 'lon']
lat = df.at[5, 'lat']
iflux = df.at[5, 'iflux']

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
center_vec = np.array(center_vec).astype(np.float32)
print(f'{center_vec=}')

ipix_fit = hp.query_disc(nside=nside, vec=center_vec, radius=2 * np.deg2rad(beam)/60)

# m_fit = np.ones(hp.nside2npix(nside))
# m_fit[ipix_fit] = 0
# hp.gnomview(m_fit, rot=[np.rad2deg(lon), np.rad2deg(lat), 0])
# plt.show()

print(f'{ipix_fit.shape=}')

def fit_model(theta, norm_beam, const):
    beam_profile = 1 / (2*np.pi*sigma**2) * np.exp(- theta**2 / (2 * sigma**2))
    # print(f'{beam_profile=}')
    return norm_beam * beam_profile + const

vec_around = hp.pix2vec(nside=nside, ipix=ipix_fit.astype(int))
print(f'{vec_around=}')
theta = np.arccos(np.array(center_vec) @ np.array(vec_around))
theta = np.nan_to_num(theta)
# theta = np.sqrt(2*(1-np.array(center_vec) @ np.array(vec_around)))

print(f'{theta=}')

y_arr = m[ipix_fit]
print(f'{y_arr}')
y_err = noise_nstd[ipix_fit]
print(f'{y_err}')
# plt.plot(ipix_fit, y_arr)
# plt.show()

lsq = LeastSquares(x=theta, y=y_arr, yerror=y_err, model=fit_model)

obj_minuit = Minuit(lsq, norm_beam=1, const=0)
obj_minuit.limits = [(0,10),(-100,100)]
# print(obj_minuit.scan(ncall=100))
# obj_minuit.errors = (0.1, 0.2)
print(obj_minuit.migrad())
print(obj_minuit.hesse())


fit_res = fit_model(theta, 1.589, 0)

plt.plot(theta, fit_res, label='hand')
plt.plot(theta, m[ipix_fit], label='true')
plt.legend()
plt.show()





