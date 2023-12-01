import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares

beam = 63 # arcmin
sigma_true = np.deg2rad(beam)/60 / (np.sqrt(8*np.log(2)))
print(f'{sigma_true=}')

nside = 512

m = np.load('../../FGSim/PSNOISE/40.npy')[0]
nstd = np.load('../../FGSim/NSTDNORTH/40.npy')[0]

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

# see_true_map()


def lsq(lon_bias1, lat_bias1, lon_bias2, lat_bias2, norm_beam1, norm_beam2, const):
    lon_fit1 = lon + lon_bias1
    lat_fit1 = lat + lat_bias1
    center_ipix1 = hp.ang2pix(nside=nside, theta=np.rad2deg(lon_fit1), phi=np.rad2deg(lat_fit1), lonlat=True)
    center_vec1 = hp.pix2vec(nside=nside, ipix=center_ipix1)
    ipix_disc1 = hp.query_disc(nside=nside, vec=center_vec1, radius=1*np.deg2rad(beam)/60)

    vec_around1 = hp.pix2vec(nside=nside, ipix=ipix_disc1.astype(int))
    theta1 = np.arccos(np.array(center_vec1) @ np.array(vec_around1))
    theta1 = np.nan_to_num(theta1)

    lon_fit2 = lon + lon_bias2
    lat_fit2 = lat + lat_bias2
    center_ipix2 = hp.ang2pix(nside=nside, theta=np.rad2deg(lon_fit2), phi=np.rad2deg(lat_fit2), lonlat=True)
    center_vec2 = hp.pix2vec(nside=nside, ipix=center_ipix2)
    ipix_disc2 = hp.query_disc(nside=nside, vec=center_vec2, radius=1*np.deg2rad(beam)/60)

    vec_around2 = hp.pix2vec(nside=nside, ipix=ipix_disc2.astype(int))
    theta2 = np.arccos(np.array(center_vec2) @ np.array(vec_around2))
    theta2 = np.nan_to_num(theta2)

    def model():
        return norm_beam1 / (2*np.pi*sigma_true**2) * np.exp(- (theta1)**2 / (2 * sigma_true**2)) + norm_beam2 / (2*np.pi*sigma_true**2) * np.exp(- (theta2)**2 / (2 * sigma_true**2)) + const

    y_model = model()
    print(f'{y_model.shape=}')

    y_data = m[ipix_disc1]
    # print(f'{y_data.shape=}')
    y_data_err = nstd[ipix_disc1]

    z = (y_data - y_model) / (y_data_err)
    return np.sum(z**2)

lsq_val = lsq(lon_bias1=0, lat_bias1=0, lon_bias2=0, lat_bias2=0, norm_beam1=7.83,norm_beam2=7.83, const=0)
print(f'{lsq_val=}')

# new_center_ipix = hp.ang2pix(nside=nside, theta=np.rad2deg(lon), phi=np.rad2deg(lat), lonlat=True)
# new_center_vec = hp.pix2vec(nside=nside, ipix=new_center_ipix)
# ipix_disc = hp.query_disc(nside=nside, vec=new_center_vec, radius=1*np.deg2rad(beam)/60)
# m_ps = np.zeros(hp.nside2npix(nside))
# m_ps[new_center_ipix] = 3.5e7
# m1 = hp.smoothing(m_ps, fwhm=np.deg2rad(beam)/60, lmax=400, iter=1)
# hp.gnomview(m1, rot=[np.rad2deg(lon), np.rad2deg(lat), 0], title='fit')
# hp.gnomview(m, rot=[np.rad2deg(lon), np.rad2deg(lat), 0], title='true')
# plt.show()



obj_minuit = Minuit(lsq, lon_bias1=0, lat_bias1=0, lon_bias2=0, lat_bias2=0, norm_beam1=1, norm_beam2=1, const=0)
bias_lonlat = np.deg2rad(0.5)
# obj_minuit.limits = [(-bias_lonlat,bias_lonlat),(-bias_lonlat,bias_lonlat),(1e5,1e8),(-100,100),(200,2000)]
obj_minuit.limits = [(-bias_lonlat,bias_lonlat),(-bias_lonlat,bias_lonlat),(-bias_lonlat,bias_lonlat),(-bias_lonlat,bias_lonlat),(0,10),(0,10),(-100,100)]
# obj_minuit.limits = [(-0.5,0.5),(-0.5,0.5),(0,10),(-100,100)]
# print(obj_minuit.scan(ncall=100))
# obj_minuit.errors = (0.1, 0.2)
print(obj_minuit.migrad())
print(obj_minuit.hesse())
ndof = 1051
str_chi2 = f"𝜒²/ndof = {obj_minuit.fval:.2f} / {ndof} = {obj_minuit.fval/ndof}"
print(str_chi2)


# fit_res = fit_model(theta, 7.83527, 0.0079154, 0)

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





