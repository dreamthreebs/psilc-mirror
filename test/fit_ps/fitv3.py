import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares

beam = 63 # arcmin
sigma = np.deg2rad(beam)/60 / (np.sqrt(8*np.log(2)))
print(f'{sigma=}')

nside = 2048

m = np.load('../../FGSim/STRPSCMBFGNOISE/40.npy')[0]
# m = np.load('../../FGSim/STRPSFGNOISE/40.npy')[0]
# m = np.load('../../FGSim/STRPSCMBNOISE/40.npy')[0]
# m = np.load('../../FGSim/PSNOISE/2048/40.npy')[0]
nstd = np.load('../../FGSim/NSTDNORTH/2048/40.npy')[0]
cstd = np.ones(hp.nside2npix(nside)) *  75.2896

df = pd.read_csv('../ps_sort/sort_by_iflux/40.csv')
lon = df.at[44, 'lon']
lat = df.at[44, 'lat']
iflux = df.at[44, 'iflux']

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

see_true_map()

center_pix = hp.ang2pix(nside=nside, theta=np.rad2deg(lon), phi=np.rad2deg(lat), lonlat=True)
# center_pix = 100000
print(f'{center_pix=}')
center_vec = hp.pix2vec(nside=nside, ipix=center_pix)
center_vec = np.array(center_vec).astype(np.float64)
print(f'{center_vec=}')

ipix_fit = hp.query_disc(nside=nside, vec=center_vec, radius=0.3 * np.deg2rad(beam)/60)
print(f'{ipix_fit.shape=}')

lon_fit = df.at[44, 'lon']
lat_fit = df.at[44, 'lat']

vec_around = hp.pix2vec(nside=nside, ipix=ipix_fit.astype(int))
theta = np.arccos(center_vec @ np.array(vec_around))
theta = np.nan_to_num(theta)

cov = np.load('./cmb_cov_data/lmax1000rf0.3.npy')
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

obj_minuit = Minuit(lsq, norm_beam=1,  const=0)
obj_minuit.limits = [(0,10),(-1e4,1e4)]
# print(obj_minuit.scan(ncall=100))
# obj_minuit.errors = (0.1, 0.2)
print(obj_minuit.migrad())
print(obj_minuit.hesse())
ndof = len(ipix_fit)
str_chi2 = f"𝜒²/ndof = {obj_minuit.fval:.2f} / {ndof} = {obj_minuit.fval/ndof}"
print(str_chi2)


fit_res = model1(theta, 0.3 , 10)

new_m = np.zeros(hp.nside2npix(nside))
new_m[ipix_fit] = fit_res
true_m = np.zeros(hp.nside2npix(nside))
true_m[ipix_fit] = m[ipix_fit]
hp.gnomview(new_m, rot=[np.rad2deg(lon), np.rad2deg(lat), 0])
hp.gnomview(true_m, rot=[np.rad2deg(lon), np.rad2deg(lat), 0])
hp.gnomview(true_m-new_m, rot=[np.rad2deg(lon), np.rad2deg(lat), 0], title='residual')
plt.show()



# plt.plot(theta, fit_res, label='hand')
# plt.plot(theta, m[ipix_fit], label='true')
# plt.legend()
# plt.show()





