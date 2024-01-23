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

class FitPointSource:
    def __init__(self, m, nstd, flux_idx, df_mask, df_ps, cl_cmb, lon, lat, iflux, lmax, nside, radius_factor, beam, sigma_threshold=5, epsilon=1e-4):
        self.m = m # sky maps (npix,)
        self.lon = lon
        self.lat = lat
        print(f'{lon=}')
        print(f'{lat=}')
        self.iflux = iflux # temperature flux in muK_CMB
        self.df_mask = df_mask # pandas data frame of point sources in mask
        self.flux_idx = flux_idx # index in df_mask
        self.df_ps = df_ps # pandas data frame of all point sources
        self.nstd = nstd # noise standard deviation
        self.cl_cmb = cl_cmb # power spectrum of CMB
        self.lmax = lmax # maximum multipole
        self.nside = nside # resolution of healpy maps
        self.radius_factor = radius_factor # disc radius of fitting region
        self.sigma_threshold = sigma_threshold # judge if a signal is a point source
        self.beam = beam # arcmin
        self.epsilon = epsilon # if CMB covariance matrix is not semi-positive, add this to cross term

        self.ini_norm_beam = self.flux2norm_beam(self.iflux)

        self.sigma = np.deg2rad(beam) / 60 / (np.sqrt(8 * np.log(2)))

        ctr0_pix = hp.ang2pix(nside=self.nside, theta=self.lon, phi=self.lat, lonlat=True)
        ctr0_vec = np.array(hp.pix2vec(nside=self.nside, ipix=ctr0_pix)).astype(np.float64)

        self.ipix_fit = hp.query_disc(nside=self.nside, vec=ctr0_vec, radius=self.radius_factor * np.deg2rad(self.beam) / 60)
        self.vec_around = np.array(hp.pix2vec(nside=self.nside, ipix=self.ipix_fit.astype(int))).astype(np.float64)
        # print(f'{ipix_fit.shape=}')
        self.ndof = len(self.ipix_fit)

        self.flag_too_near = False

    def flux2norm_beam(self, flux):
        # from mJy to muK_CMB to norm_beam
        coeffmJy2norm = 2.1198465131100624e-05
        return coeffmJy2norm * flux

    def input_lonlat2pix_lonlat(self, input_lon, input_lat):
        ipix = hp.ang2pix(nside=self.nside, theta=input_lon, phi=input_lat, lonlat=True)
        out_lon, out_lat = hp.pix2ang(nside=self.nside, ipix=ipix, lonlat=True)
        return out_lon, out_lat

    def change_lonlat2pixlonlat(self):
        lon = np.rad2deg(self.df_ps.loc[:,'lon'].to_numpy())
        lat = np.rad2deg(self.df_ps.loc[:,'lat'].to_numpy())
        print(f'{lon.shape=}')
        ipix = hp.ang2pix(nside=self.nside, theta=lon, phi=lat, lonlat=True)
        pix_lon, pix_lat = hp.pix2ang(nside=self.nside, ipix=ipix, lonlat=True)
        pix_lon_rad = np.deg2rad(pix_lon)
        pix_lat_rad = np.deg2rad(pix_lat)

        diff_lon = pix_lon - lon
        diff_lat = pix_lat - lat
        print(f'{diff_lon=}')
        print(f'{diff_lat=}')
        print(f'{np.max(np.abs(diff_lon))=}')
        print(f'{np.max(np.abs(diff_lat))=}')


        df_ps['lon'] = pix_lon_rad
        df_ps['lat'] = pix_lat_rad
        df_ps.to_csv('./40ps.csv')




if __name__ == '__main__':
    m = np.load('../../../../FGSim/FITDATA/PSNOISE/40.npy')[0]
    # m = np.load('../../FGSim/FITDATA/PSCMBNOISE/40.npy')[0]
    nstd = np.load('../../../../FGSim/NSTDNORTH/2048/40.npy')[0]
    df_mask = pd.read_csv('../../../partial_sky_ps/ps_in_mask/mask40.csv')
    flux_idx = 1
    lon = np.rad2deg(df_mask.at[flux_idx, 'lon'])
    lat = np.rad2deg(df_mask.at[flux_idx, 'lat'])
    iflux = df_mask.at[flux_idx, 'iflux']

    df_ps = pd.read_csv('../../../../test/ps_sort/sort_by_iflux/40.csv')
    
    lmax = 350
    nside = 2048
    beam = 63
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax)
    # m = np.load('../../inpaintingdata/CMB8/40.npy')[0]
    # cl1 = hp.anafast(m, lmax=lmax)
    cl_cmb = np.load('../../../../src/cmbsim/cmbdata/cmbcl.npy')[:lmax+1,0]
    # l = np.arange(lmax+1)

    # plt.plot(l*(l+1)*cl_cmb/(2*np.pi))
    cl_cmb = cl_cmb * bl**2

    # plt.plot(l*(l+1)*cl_cmb/(2*np.pi))
    # plt.plot(l*(l+1)*cl1/(2*np.pi), label='cl1')
    # plt.show()

    obj = FitPointSource(m=m, nstd=nstd, flux_idx=flux_idx, df_mask=df_mask, df_ps=df_ps, cl_cmb=cl_cmb, lon=lon, lat=lat, iflux=iflux, lmax=lmax, nside=nside, radius_factor=2.0, beam=beam, epsilon=1e-4)

    # obj.see_true_map(m=m, lon=lon, lat=lat, nside=nside, beam=beam)

    obj.change_lonlat2pixlonlat()


