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

class FitPointSource:
    def __init__(self, m, nstd, flux_idx, df_mask, df_ps, cl_cmb, lon, lat, iflux, lmax, nside, radius_factor, beam, sigma_threshold=5):
        self.m = m # sky maps (npix,)
        self.input_lon = lon # input longitude in degrees
        self.input_lat = lat # input latitude in degrees
        ipix = hp.ang2pix(nside=nside, theta=lon, phi=lat, lonlat=True)
        lon, lat = hp.pix2ang(nside=nside, ipix=ipix, lonlat=True)
        self.lon = lon
        self.lat = lat
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

        self.n_pts = 0 # count number for counting
        self.n_call = 0.0 # count number of function calls

        self.sigma = np.deg2rad(beam) / 60 / (np.sqrt(8 * np.log(2)))

    def flux2norm_beam(self, flux):
        # from mJy to muK_CMB to norm_beam
        coeffmJy2norm = 2.1198465131100624e-05
        return coeffmJy2norm * flux

    def input_lonlat2pix_lonlat(self, input_lon, input_lat):
        ipix = hp.ang2pix(nside=self.nside, theta=input_lon, phi=input_lat, lonlat=True)
        out_lon, out_lat = hp.pix2ang(nside=self.nside, ipix=ipix, lonlat=True)
        return out_lon, out_lat

    def see_true_map(self, m, lon, lat, nside, beam):
        radiops = hp.read_map('/sharefs/alicpt/users/zrzhang/allFreqPSMOutput/skyinbands/AliCPT_uKCMB/40GHz/strongradiops_map_40GHz.fits', field=0)
        irps = hp.read_map('/sharefs/alicpt/users/zrzhang/allFreqPSMOutput/skyinbands/AliCPT_uKCMB/40GHz/strongirps_map_40GHz.fits', field=0)

        hp.gnomview(irps, rot=[lon, lat, 0], xsize=300, ysize=300, reso=1, title='irps')
        hp.gnomview(radiops, rot=[lon, lat, 0], xsize=300, ysize=300, reso=1, title='radiops')
        hp.gnomview(m, rot=[lon, lat, 0], xsize=300, ysize=300)
        plt.show()

        vec = hp.ang2vec(theta=lon, phi=lat, lonlat=True)
        ipix_disc = hp.query_disc(nside=nside, vec=vec, radius=np.deg2rad(beam)/60)

        mask = np.ones(hp.nside2npix(nside))
        mask[ipix_disc] = 0

        hp.gnomview(mask, rot=[lon, lat, 0])
        plt.show()

    def find_nearby_ps_lon_lat(self, threshold_factor=2.2):
        dir_0 = (self.lon, self.lat)
        arr_1 = self.df_ps.loc[:,'Unnamed: 0']
        print(f'{arr_1.shape=}')
        bool_arr = self.df_ps.loc[:, 'Unnamed: 0'] != self.df_mask.at[self.flux_idx, 'Unnamed: 0']
        print(f'{bool_arr.shape=}')
        lon_other = np.rad2deg(self.df_ps.loc[bool_arr, 'lon'])
        lat_other = np.rad2deg(self.df_ps.loc[bool_arr, 'lat'])
        dir_other = (lon_other, lat_other)
        ang = np.rad2deg(hp.rotator.angdist(dir1=dir_0, dir2=dir_other, lonlat=True))
        print(f'{ang.shape=}')
        threshold = threshold_factor * self.beam / 60
        print(f'{ang=}')

        index_near = np.nonzero(np.where((ang < threshold), ang, 0))
        ang_near = ang[index_near]
        print(f'{index_near=}')
        print(f'{ang_near=}')
        if index_near[0].size == 0:
            raise ValueError('This is a single point source, please check! 4 parameter fit should get good fitting result')
        else:
            if index_near[0][0] < self.df_mask.at[self.flux_idx,'Unnamed: 0']:
                lon = np.rad2deg(self.df_ps.at[index_near[0][0], 'lon'])
                lat = np.rad2deg(self.df_ps.at[index_near[0][0], 'lat'])
                lon, lat = self.input_lonlat2pix_lonlat(lon, lat)
                iflux = self.flux2norm_beam(self.df_ps.at[index_near[0][0], 'iflux'])
            else:
                lon = np.rad2deg(self.df_ps.at[index_near[0][0]+1, 'lon'])
                lat = np.rad2deg(self.df_ps.at[index_near[0][0]+1, 'lat'])
                lon, lat = self.input_lonlat2pix_lonlat(lon, lat)
                iflux = self.flux2norm_beam(self.df_ps.at[index_near[0][0]+1, 'iflux'])
            print(f'{iflux=}')

            # hp.gnomview(self.m, rot=[self.lon,self.lat,0])
            # hp.projscatter(self.lon, self.lat, lonlat=True)
            # hp.projscatter(lon, lat, lonlat=True)
            # plt.show()

            return iflux, lon, lat
    def find_first_second_nearby_ps_lon_lat(self, threshold_factor=2.2):
        dir_0 = (self.lon, self.lat)
        arr_1 = self.df_ps.loc[:,'Unnamed: 0']
        print(f'{arr_1.shape=}')
        bool_arr = self.df_ps.loc[:, 'Unnamed: 0'] != self.df_mask.at[self.flux_idx, 'Unnamed: 0']
        print(f'{bool_arr.shape=}')
        lon_other = np.rad2deg(self.df_ps.loc[bool_arr, 'lon'])
        lat_other = np.rad2deg(self.df_ps.loc[bool_arr, 'lat'])
        dir_other = (lon_other, lat_other)
        ang = np.rad2deg(hp.rotator.angdist(dir1=dir_0, dir2=dir_other, lonlat=True))
        print(f'{ang.shape=}')
        threshold = threshold_factor * self.beam / 60
        print(f'{ang=}')

        index_near = np.nonzero(np.where((ang < threshold), ang, 0))
        ang_near = ang[index_near]
        print(f'{index_near=}')
        print(f'{ang_near=}')
        if index_near[0].size == 0:
            raise ValueError('This is a single point source, please check! 4 parameter fit should get good fitting result')
        else:
            lon_list = []
            lat_list = []
            iflux_list = []
            condition = index_near[0] < self.df_mask.at[self.flux_idx,'Unnamed: 0']
            if np.any(condition):
                n_smaller = np.count_nonzero(condition)
                for i in range(n_smaller):
                    lon = np.rad2deg(self.df_ps.at[index_near[0][i], 'lon'])
                    lat = np.rad2deg(self.df_ps.at[index_near[0][i], 'lat'])
                    lon, lat = self.input_lonlat2pix_lonlat(lon, lat)
                    iflux = self.flux2norm_beam(self.df_ps.at[index_near[0][i], 'iflux'])
                    lon_list.append(lon)
                    lat_list.append(lat)
                    iflux_list.append(iflux)
                for i in range(n_smaller, np.size(condition)):
                    lon = np.rad2deg(self.df_ps.at[index_near[0][i]+1, 'lon'])
                    lat = np.rad2deg(self.df_ps.at[index_near[0][i]+1, 'lat'])
                    lon, lat = self.input_lonlat2pix_lonlat(lon, lat)
                    iflux = self.flux2norm_beam(self.df_ps.at[index_near[0][i]+1, 'iflux'])
                    lon_list.append(lon)
                    lat_list.append(lat)
                    iflux_list.append(iflux)
            else:
                n_nearby_pts = np.size(condition)
                for i in range(n_nearby_pts):
                    lon = np.rad2deg(self.df_ps.at[index_near[0][i]+1, 'lon'])
                    lat = np.rad2deg(self.df_ps.at[index_near[0][i]+1, 'lat'])
                    lon, lat = self.input_lonlat2pix_lonlat(lon, lat)
                    iflux = self.flux2norm_beam(self.df_ps.at[index_near[0][i]+1, 'iflux'])
                    lon_list.append(lon)
                    lat_list.append(lat)
                    iflux_list.append(iflux)
            print(f'{iflux_list=}')

            hp.gnomview(self.m, rot=[self.lon,self.lat,0])
            # hp.projscatter(self.lon, self.lat, lonlat=True)
            hp.projscatter(lon_list[0], lat_list[0], lonlat=True)
            hp.projscatter(lon_list[1], lat_list[1], lonlat=True)
            plt.show()

            return iflux_list[0], lon_list[0], lat_list[0], iflux_list[1], lon_list[1], lat_list[1]
    def find_first_second_third_nearby_ps_lon_lat(self, threshold_factor=2.2):
        dir_0 = (self.lon, self.lat)
        arr_1 = self.df_ps.loc[:,'Unnamed: 0']
        print(f'{arr_1.shape=}')
        bool_arr = self.df_ps.loc[:, 'Unnamed: 0'] != self.df_mask.at[self.flux_idx, 'Unnamed: 0']
        print(f'{bool_arr.shape=}')
        lon_other = np.rad2deg(self.df_ps.loc[bool_arr, 'lon'])
        lat_other = np.rad2deg(self.df_ps.loc[bool_arr, 'lat'])
        dir_other = (lon_other, lat_other)
        ang = np.rad2deg(hp.rotator.angdist(dir1=dir_0, dir2=dir_other, lonlat=True))
        print(f'{ang.shape=}')
        threshold = threshold_factor * self.beam / 60
        print(f'{ang=}')

        index_near = np.nonzero(np.where((ang < threshold), ang, 0))
        ang_near = ang[index_near]
        print(f'{index_near=}')
        print(f'{ang_near=}')
        if index_near[0].size == 0:
            raise ValueError('This is a single point source, please check! 4 parameter fit should get good fitting result')
        else:
            lon_list = []
            lat_list = []
            iflux_list = []
            condition = index_near[0] < self.df_mask.at[self.flux_idx,'Unnamed: 0']
            if np.any(condition):
                n_smaller = np.count_nonzero(condition)
                for i in range(n_smaller):
                    lon = np.rad2deg(self.df_ps.at[index_near[0][i], 'lon'])
                    lat = np.rad2deg(self.df_ps.at[index_near[0][i], 'lat'])
                    lon, lat = self.input_lonlat2pix_lonlat(lon, lat)
                    iflux = self.flux2norm_beam(self.df_ps.at[index_near[0][i], 'iflux'])
                    lon_list.append(lon)
                    lat_list.append(lat)
                    iflux_list.append(iflux)
                for i in range(n_smaller, np.size(condition)):
                    lon = np.rad2deg(self.df_ps.at[index_near[0][i]+1, 'lon'])
                    lat = np.rad2deg(self.df_ps.at[index_near[0][i]+1, 'lat'])
                    lon, lat = self.input_lonlat2pix_lonlat(lon, lat)
                    iflux = self.flux2norm_beam(self.df_ps.at[index_near[0][i]+1, 'iflux'])
                    lon_list.append(lon)
                    lat_list.append(lat)
                    iflux_list.append(iflux)

            else:
                n_nearby_pts = np.size(condition)
                for i in range(n_nearby_pts):
                    lon = np.rad2deg(self.df_ps.at[index_near[0][i]+1, 'lon'])
                    lat = np.rad2deg(self.df_ps.at[index_near[0][i]+1, 'lat'])
                    lon, lat = self.input_lonlat2pix_lonlat(lon, lat)
                    iflux = self.flux2norm_beam(self.df_ps.at[index_near[0][i]+1, 'iflux'])
                    lon_list.append(lon)
                    lat_list.append(lat)
                    iflux_list.append(iflux)
            print(f'{iflux_list=}')

            # hp.gnomview(self.m, rot=[self.lon,self.lat,0])
            # # hp.projscatter(self.lon, self.lat, lonlat=True)
            # hp.projscatter(lon_list[0], lat_list[0], lonlat=True)
            # hp.projscatter(lon_list[1], lat_list[1], lonlat=True)
            # plt.show()

            return iflux_list[0], lon_list[0], lat_list[0], iflux_list[1], lon_list[1], lat_list[1], iflux_list[2], lon_list[2], lat_list[2]
    def find_first_second_third_fourth_nearby_ps_lon_lat(self, threshold_factor=2.2):
        dir_0 = (self.lon, self.lat)
        arr_1 = self.df_ps.loc[:,'Unnamed: 0']
        print(f'{arr_1.shape=}')
        bool_arr = self.df_ps.loc[:, 'Unnamed: 0'] != self.df_mask.at[self.flux_idx, 'Unnamed: 0']
        print(f'{bool_arr.shape=}')
        lon_other = np.rad2deg(self.df_ps.loc[bool_arr, 'lon'])
        lat_other = np.rad2deg(self.df_ps.loc[bool_arr, 'lat'])
        dir_other = (lon_other, lat_other)
        ang = np.rad2deg(hp.rotator.angdist(dir1=dir_0, dir2=dir_other, lonlat=True))
        print(f'{ang.shape=}')
        threshold = threshold_factor * self.beam / 60
        print(f'{ang=}')

        index_near = np.nonzero(np.where((ang < threshold), ang, 0))
        ang_near = ang[index_near]
        print(f'{index_near=}')
        print(f'{ang_near=}')
        if index_near[0].size == 0:
            raise ValueError('This is a single point source, please check! 4 parameter fit should get good fitting result')
        else:
            lon_list = []
            lat_list = []
            iflux_list = []
            condition = index_near[0] < self.df_mask.at[self.flux_idx,'Unnamed: 0']
            if np.any(condition):
                n_smaller = np.count_nonzero(condition)
                for i in range(n_smaller):
                    lon = np.rad2deg(self.df_ps.at[index_near[0][i], 'lon'])
                    lat = np.rad2deg(self.df_ps.at[index_near[0][i], 'lat'])
                    lon, lat = self.input_lonlat2pix_lonlat(lon, lat)
                    iflux = self.flux2norm_beam(self.df_ps.at[index_near[0][i], 'iflux'])
                    lon_list.append(lon)
                    lat_list.append(lat)
                    iflux_list.append(iflux)
                for i in range(n_smaller, np.size(condition)):
                    lon = np.rad2deg(self.df_ps.at[index_near[0][i]+1, 'lon'])
                    lat = np.rad2deg(self.df_ps.at[index_near[0][i]+1, 'lat'])
                    lon, lat = self.input_lonlat2pix_lonlat(lon, lat)
                    iflux = self.flux2norm_beam(self.df_ps.at[index_near[0][i]+1, 'iflux'])
                    lon_list.append(lon)
                    lat_list.append(lat)
                    iflux_list.append(iflux)

            else:
                n_nearby_pts = np.size(condition)
                for i in range(n_nearby_pts):
                    lon = np.rad2deg(self.df_ps.at[index_near[0][i]+1, 'lon'])
                    lat = np.rad2deg(self.df_ps.at[index_near[0][i]+1, 'lat'])
                    lon, lat = self.input_lonlat2pix_lonlat(lon, lat)
                    iflux = self.flux2norm_beam(self.df_ps.at[index_near[0][i]+1, 'iflux'])
                    lon_list.append(lon)
                    lat_list.append(lat)
                    iflux_list.append(iflux)

            print(f'{iflux_list=}')
            # hp.gnomview(self.m, rot=[self.lon,self.lat,0])
            # # hp.projscatter(self.lon, self.lat, lonlat=True)
            # hp.projscatter(lon_list[0], lat_list[0], lonlat=True)
            # hp.projscatter(lon_list[1], lat_list[1], lonlat=True)
            # plt.show()

            return iflux_list[0], lon_list[0], lat_list[0], iflux_list[1], lon_list[1], lat_list[1], iflux_list[2], lon_list[2], lat_list[2], iflux_list[3], lon_list[3], lat_list[3]


    def fit_ps_ns(self, mode:str='pipeline'):
        def lsq_2_params(norm_beam, const):
            ctr_lon = self.lon
            ctr_lat = self.lat

            ctr_pix = hp.ang2pix(nside=self.nside, theta=ctr_lon, phi=ctr_lat, lonlat=True)
            ctr_vec = np.array(hp.pix2vec(nside=self.nside, ipix=ctr_pix)).astype(np.float64)

            ipix_fit = hp.query_disc(nside=self.nside, vec=ctr_vec, radius=self.radius_factor * np.deg2rad(self.beam) / 60)
            # print(f'{ipix_fit.shape=}')
            self.lsq_2_ndof = len(ipix_fit)
            vec_around = np.array(hp.pix2vec(nside=self.nside, ipix=ipix_fit.astype(int))).astype(np.float64)

            theta = hp.rotator.angdist(dir1=ctr_vec, dir2=vec_around)

            def model():
                return norm_beam / (2 * np.pi * self.sigma**2) * np.exp(- (theta)**2 / (2 * self.sigma**2)) + const

            y_model = model()
            y_data = self.m[ipix_fit]
            y_err = self.nstd[ipix_fit]

            z = (y_data - y_model) / y_err
            return np.sum(z**2)

        def lsq_4_params(norm_beam, ctr_lon_shift, ctr_lat_shift, const):
            ctr_lon = self.lon + ctr_lon_shift
            ctr_lat = self.lat + ctr_lat_shift

            ctr_pix = hp.ang2pix(nside=self.nside, theta=ctr_lon, phi=ctr_lat, lonlat=True)
            ctr_vec = np.array(hp.pix2vec(nside=self.nside, ipix=ctr_pix)).astype(np.float64)

            ipix_fit = hp.query_disc(nside=self.nside, vec=ctr_vec, radius=self.radius_factor * np.deg2rad(self.beam) / 60)
            # print(f'{ipix_fit.shape=}')
            vec_around = np.array(hp.pix2vec(nside=self.nside, ipix=ipix_fit.astype(int))).astype(np.float64)

            theta = hp.rotator.angdist(dir1=ctr_vec, dir2=vec_around)

            def model():
                return norm_beam / (2 * np.pi * self.sigma**2) * np.exp(- (theta)**2 / (2 * self.sigma**2)) + const

            y_model = model()
            y_data = self.m[ipix_fit]
            y_err = self.nstd[ipix_fit]

            z = (y_data - y_model) / y_err
            self.n_pts += len(ipix_fit)
            self.n_call += 1
            return np.sum(z**2)

        def lsq_7_params(norm_beam1, ctr1_lon_shift, ctr1_lat_shift, norm_beam2, ctr2_lon_shift, ctr2_lat_shift, const):
            ctr1_lon = self.lon + ctr1_lon_shift
            ctr1_lat = self.lat + ctr1_lat_shift

            ctr2_lon = self.ctr2_lon + ctr2_lon_shift
            ctr2_lat = self.ctr2_lat + ctr2_lat_shift

            ctr1_pix = hp.ang2pix(nside=self.nside, theta=ctr1_lon, phi=ctr1_lat, lonlat=True)
            ctr1_vec = np.array(hp.pix2vec(nside=self.nside, ipix=ctr1_pix)).astype(np.float64)
            ctr2_pix = hp.ang2pix(nside=self.nside, theta=ctr2_lon, phi=ctr2_lat, lonlat=True)
            ctr2_vec = np.array(hp.pix2vec(nside=self.nside, ipix=ctr2_pix)).astype(np.float64)

            ipix_fit = hp.query_disc(nside=self.nside, vec=ctr1_vec, radius=self.radius_factor * np.deg2rad(self.beam) / 60)
            # print(f'{ipix_fit.shape=}')
            vec_around = np.array(hp.pix2vec(nside=self.nside, ipix=ipix_fit.astype(int))).astype(np.float64)

            theta1 = hp.rotator.angdist(dir1=ctr1_vec, dir2=vec_around)
            theta2 = hp.rotator.angdist(dir1=ctr2_vec, dir2=vec_around)

            def model():
                return norm_beam1 / (2 * np.pi * self.sigma**2) * np.exp(- (theta1)**2 / (2 * self.sigma**2)) + norm_beam2 / (2 * np.pi * self.sigma**2) * np.exp(- (theta2)**2 / (2 * self.sigma**2)) + const

            y_model = model()
            y_data = self.m[ipix_fit]
            y_err = self.nstd[ipix_fit]

            z = (y_data - y_model) / y_err
            self.n_pts += len(ipix_fit)
            self.n_call += 1
            return np.sum(z**2)
        def lsq_10_params(norm_beam1, ctr1_lon_shift, ctr1_lat_shift, norm_beam2, ctr2_lon_shift, ctr2_lat_shift, norm_beam3, ctr3_lon_shift, ctr3_lat_shift, const):
            ctr1_lon = self.lon + ctr1_lon_shift
            ctr1_lat = self.lat + ctr1_lat_shift
            ctr2_lon = self.ctr2_lon + ctr2_lon_shift
            ctr2_lat = self.ctr2_lat + ctr2_lat_shift
            ctr3_lon = self.ctr3_lon + ctr3_lon_shift
            ctr3_lat = self.ctr3_lat + ctr3_lat_shift

            ctr1_pix = hp.ang2pix(nside=self.nside, theta=ctr1_lon, phi=ctr1_lat, lonlat=True)
            ctr1_vec = np.array(hp.pix2vec(nside=self.nside, ipix=ctr1_pix)).astype(np.float64)
            ctr2_pix = hp.ang2pix(nside=self.nside, theta=ctr2_lon, phi=ctr2_lat, lonlat=True)
            ctr2_vec = np.array(hp.pix2vec(nside=self.nside, ipix=ctr2_pix)).astype(np.float64)
            ctr3_pix = hp.ang2pix(nside=self.nside, theta=ctr3_lon, phi=ctr3_lat, lonlat=True)
            ctr3_vec = np.array(hp.pix2vec(nside=self.nside, ipix=ctr3_pix)).astype(np.float64)

            ipix_fit = hp.query_disc(nside=self.nside, vec=ctr1_vec, radius=self.radius_factor * np.deg2rad(self.beam) / 60)
            # print(f'{ipix_fit.shape=}')
            vec_around = np.array(hp.pix2vec(nside=self.nside, ipix=ipix_fit.astype(int))).astype(np.float64)

            theta1 = hp.rotator.angdist(dir1=ctr1_vec, dir2=vec_around)
            theta2 = hp.rotator.angdist(dir1=ctr2_vec, dir2=vec_around)
            theta3 = hp.rotator.angdist(dir1=ctr3_vec, dir2=vec_around)

            def model():
                return norm_beam1 / (2 * np.pi * self.sigma**2) * np.exp(- (theta1)**2 / (2 * self.sigma**2)) + norm_beam2 / (2 * np.pi * self.sigma**2) * np.exp(- (theta2)**2 / (2 * self.sigma**2)) + norm_beam3 / (2 * np.pi * self.sigma**2) * np.exp(- (theta3)**2 / (2 * self.sigma**2)) + const

            y_model = model()
            y_data = self.m[ipix_fit]
            y_err = self.nstd[ipix_fit]

            z = (y_data - y_model) / y_err
            self.n_pts += len(ipix_fit)
            self.n_call += 1
            return np.sum(z**2)
        def lsq_13_params(norm_beam1, ctr1_lon_shift, ctr1_lat_shift, norm_beam2, ctr2_lon_shift, ctr2_lat_shift, norm_beam3, ctr3_lon_shift, ctr3_lat_shift, norm_beam4, ctr4_lon_shift, ctr4_lat_shift,const):
            ctr1_lon = self.lon + ctr1_lon_shift
            ctr1_lat = self.lat + ctr1_lat_shift
            ctr2_lon = self.ctr2_lon + ctr2_lon_shift
            ctr2_lat = self.ctr2_lat + ctr2_lat_shift
            ctr3_lon = self.ctr3_lon + ctr3_lon_shift
            ctr3_lat = self.ctr3_lat + ctr3_lat_shift
            ctr4_lon = self.ctr4_lon + ctr4_lon_shift
            ctr4_lat = self.ctr4_lat + ctr4_lat_shift

            ctr1_pix = hp.ang2pix(nside=self.nside, theta=ctr1_lon, phi=ctr1_lat, lonlat=True)
            ctr1_vec = np.array(hp.pix2vec(nside=self.nside, ipix=ctr1_pix)).astype(np.float64)
            ctr2_pix = hp.ang2pix(nside=self.nside, theta=ctr2_lon, phi=ctr2_lat, lonlat=True)
            ctr2_vec = np.array(hp.pix2vec(nside=self.nside, ipix=ctr2_pix)).astype(np.float64)
            ctr3_pix = hp.ang2pix(nside=self.nside, theta=ctr3_lon, phi=ctr3_lat, lonlat=True)
            ctr3_vec = np.array(hp.pix2vec(nside=self.nside, ipix=ctr3_pix)).astype(np.float64)
            ctr4_pix = hp.ang2pix(nside=self.nside, theta=ctr4_lon, phi=ctr4_lat, lonlat=True)
            ctr4_vec = np.array(hp.pix2vec(nside=self.nside, ipix=ctr4_pix)).astype(np.float64)

            ipix_fit = hp.query_disc(nside=self.nside, vec=ctr1_vec, radius=self.radius_factor * np.deg2rad(self.beam) / 60)
            # print(f'{ipix_fit.shape=}')
            vec_around = np.array(hp.pix2vec(nside=self.nside, ipix=ipix_fit.astype(int))).astype(np.float64)

            theta1 = hp.rotator.angdist(dir1=ctr1_vec, dir2=vec_around)
            theta2 = hp.rotator.angdist(dir1=ctr2_vec, dir2=vec_around)
            theta3 = hp.rotator.angdist(dir1=ctr3_vec, dir2=vec_around)
            theta4 = hp.rotator.angdist(dir1=ctr4_vec, dir2=vec_around)

            def model():
                return norm_beam1 / (2 * np.pi * self.sigma**2) * np.exp(- (theta1)**2 / (2 * self.sigma**2)) + norm_beam2 / (2 * np.pi * self.sigma**2) * np.exp(- (theta2)**2 / (2 * self.sigma**2)) + norm_beam3 / (2 * np.pi * self.sigma**2) * np.exp(- (theta3)**2 / (2 * self.sigma**2)) + norm_beam4 / (2 * np.pi * self.sigma**2) * np.exp(- (theta4)**2 / (2 * self.sigma**2)) + const

            y_model = model()
            y_data = self.m[ipix_fit]
            y_err = self.nstd[ipix_fit]

            z = (y_data - y_model) / y_err
            self.n_pts += len(ipix_fit)
            self.n_call += 1
            return np.sum(z**2)
        def lsq_16_params(norm_beam1, ctr1_lon_shift, ctr1_lat_shift, norm_beam2, ctr2_lon_shift, ctr2_lat_shift, norm_beam3, ctr3_lon_shift, ctr3_lat_shift, norm_beam4, ctr4_lon_shift, ctr4_lat_shift, norm_beam5, ctr5_lon_shift, ctr5_lat_shift, const):
            ctr1_lon = self.lon + ctr1_lon_shift
            ctr1_lat = self.lat + ctr1_lat_shift
            ctr2_lon = self.ctr2_lon + ctr2_lon_shift
            ctr2_lat = self.ctr2_lat + ctr2_lat_shift
            ctr3_lon = self.ctr3_lon + ctr3_lon_shift
            ctr3_lat = self.ctr3_lat + ctr3_lat_shift
            ctr4_lon = self.ctr4_lon + ctr4_lon_shift
            ctr4_lat = self.ctr4_lat + ctr4_lat_shift
            ctr5_lon = self.ctr5_lon + ctr5_lon_shift
            ctr5_lat = self.ctr5_lat + ctr5_lat_shift

            ctr1_pix = hp.ang2pix(nside=self.nside, theta=ctr1_lon, phi=ctr1_lat, lonlat=True)
            ctr1_vec = np.array(hp.pix2vec(nside=self.nside, ipix=ctr1_pix)).astype(np.float64)
            ctr2_pix = hp.ang2pix(nside=self.nside, theta=ctr2_lon, phi=ctr2_lat, lonlat=True)
            ctr2_vec = np.array(hp.pix2vec(nside=self.nside, ipix=ctr2_pix)).astype(np.float64)
            ctr3_pix = hp.ang2pix(nside=self.nside, theta=ctr3_lon, phi=ctr3_lat, lonlat=True)
            ctr3_vec = np.array(hp.pix2vec(nside=self.nside, ipix=ctr3_pix)).astype(np.float64)
            ctr4_pix = hp.ang2pix(nside=self.nside, theta=ctr4_lon, phi=ctr4_lat, lonlat=True)
            ctr4_vec = np.array(hp.pix2vec(nside=self.nside, ipix=ctr4_pix)).astype(np.float64)
            ctr5_pix = hp.ang2pix(nside=self.nside, theta=ctr5_lon, phi=ctr5_lat, lonlat=True)
            ctr5_vec = np.array(hp.pix2vec(nside=self.nside, ipix=ctr5_pix)).astype(np.float64)

            ipix_fit = hp.query_disc(nside=self.nside, vec=ctr1_vec, radius=self.radius_factor * np.deg2rad(self.beam) / 60)
            # print(f'{ipix_fit.shape=}')
            vec_around = np.array(hp.pix2vec(nside=self.nside, ipix=ipix_fit.astype(int))).astype(np.float64)

            theta1 = hp.rotator.angdist(dir1=ctr1_vec, dir2=vec_around)
            theta2 = hp.rotator.angdist(dir1=ctr2_vec, dir2=vec_around)
            theta3 = hp.rotator.angdist(dir1=ctr3_vec, dir2=vec_around)
            theta4 = hp.rotator.angdist(dir1=ctr4_vec, dir2=vec_around)
            theta5 = hp.rotator.angdist(dir1=ctr5_vec, dir2=vec_around)

            def model():
                return norm_beam1 / (2 * np.pi * self.sigma**2) * np.exp(- (theta1)**2 / (2 * self.sigma**2)) + norm_beam2 / (2 * np.pi * self.sigma**2) * np.exp(- (theta2)**2 / (2 * self.sigma**2)) + norm_beam3 / (2 * np.pi * self.sigma**2) * np.exp(- (theta3)**2 / (2 * self.sigma**2)) + norm_beam4 / (2 * np.pi * self.sigma**2) * np.exp(- (theta4)**2 / (2 * self.sigma**2)) + norm_beam5 / (2 * np.pi * self.sigma**2) * np.exp(- (theta5)**2 / (2 * self.sigma**2))+ const

            y_model = model()
            y_data = self.m[ipix_fit]
            y_err = self.nstd[ipix_fit]

            z = (y_data - y_model) / y_err
            self.n_pts += len(ipix_fit)
            self.n_call += 1
            return np.sum(z**2)



        if mode == 'pipeline':

            self.ini_norm_beam = self.flux2norm_beam(self.iflux)
            print(f'begin point source fitting, first do 2 parameter fit...')
            obj_minuit = Minuit(lsq_2_params, norm_beam=self.ini_norm_beam, const=0.0)
            obj_minuit.limits = [(0,1),  (-100,100)]
            print(obj_minuit.migrad())

            chi2dof = obj_minuit.fval / self.lsq_2_ndof
            str_chi2 = f"𝜒²/ndof = {obj_minuit.fval:.2f} / {self.lsq_2_ndof} = {chi2dof}"
            print(str_chi2)

            is_ps = obj_minuit.values['norm_beam'] > self.sigma_threshold * obj_minuit.errors['norm_beam']
            if obj_minuit.fmin.hesse_failed:
                raise ValueError('hesse failed!')
            elif (chi2dof <  1.04):
                if not is_ps:
                    print(f'not a point source under {self.sigma_threshold} sigma threshold')
                    return 0.0, self.lon, self.lat

            obj_minuit = Minuit(lsq_4_params, norm_beam=self.ini_norm_beam, ctr_lon_shift=0.0, ctr_lat_shift=0.0, const=0.0)

            obj_minuit.limits = [(0,1), (-0.5,0.5), (-0.5,0.5), (-100,100)]
            print(obj_minuit.migrad())
            print(obj_minuit.hesse())
            ndof = self.n_pts / (self.n_call - 4)
            chi2dof = obj_minuit.fval / ndof
            str_chi2 = f"𝜒²/ndof = {obj_minuit.fval:.2f} / {ndof} = {chi2dof}"
            print(str_chi2)
            is_ps = obj_minuit.values['norm_beam'] > self.sigma_threshold * obj_minuit.errors['norm_beam']

            if obj_minuit.fmin.hesse_failed:
                raise ValueError('hesse failed!')
            elif (chi2dof < 1.04):
                if is_ps:
                    print(f'4 parameter fitting is enough, hesse ok')
                    fit_lon = self.lon + obj_minuit.values['ctr_lon_shift']
                    fit_lat = self.lon + obj_minuit.values['ctr_lat_shift']

                    self.fit_norm_beam = obj_minuit.values['norm_beam']
                    true_norm_beam = self.flux2norm_beam(self.iflux)
                    print(f'{true_norm_beam=}')
                    print(f'fit_norm_beam={self.fit_norm_beam}')
                    return obj_minuit.values['norm_beam'], fit_lon, fit_lat
                else:
                    print(f'not a point source under {self.sigma_threshold} sigma threshold')
                    return 0.0, self.lon, self.lat
            else:
                self.do_7_params = True

            if self.do_7_params:
                print(f'4 parameter fitting is not enough, try 7 parameter fitting...')
                self.ctr2_iflux, self.ctr2_lon, self.ctr2_lat = self.find_nearby_ps_lon_lat()
                self.n_pts = 0
                self.n_call = 0
                obj_minuit = Minuit(lsq_7_params, norm_beam1=self.ini_norm_beam, ctr1_lon_shift=0.0, ctr1_lat_shift=0.0, norm_beam2=self.ctr2_iflux, ctr2_lon_shift=0.0, ctr2_lat_shift=0.0, const=0.0)
                obj_minuit.limits = [(0,1), (-0.5,0.6), (-0.5,0.6),(0,1),(-0.5,0.6),(-0.5,0.6), (-100,100)]
                print(obj_minuit.migrad())
                print(obj_minuit.hesse())
                ndof = self.n_pts / (self.n_call - 7)
                chi2dof = obj_minuit.fval / ndof
                str_chi2 = f"𝜒²/ndof = {obj_minuit.fval:.2f} / {ndof} = {chi2dof}"
                print(str_chi2)

                is_ps = obj_minuit.values['norm_beam1'] > self.sigma_threshold * obj_minuit.errors['norm_beam1']
                if obj_minuit.fmin.hesse_failed:
                    raise ValueError('hesse failed!')
                elif (chi2dof <  1.05):
                    if is_ps:
                        print(f'7 parameter fitting is enough, hesse ok')
                        fit_lon = self.lon + obj_minuit.values['ctr1_lon_shift']
                        fit_lat = self.lon + obj_minuit.values['ctr1_lat_shift']

                        self.fit_norm_beam = obj_minuit.values['norm_beam1']
                        true_norm_beam = self.flux2norm_beam(self.iflux)
                        print(f'{true_norm_beam=}')
                        print(f'fit_norm_beam={self.fit_norm_beam}')
                        return obj_minuit.values['norm_beam1'], fit_lon, fit_lat
                    else:
                        print(f'not a point source under {self.sigma_threshold} sigma threshold')
                        return 0.0, self.lon, self.lat
                else:
                    self.do_10_params = True

            if self.do_10_params:
                print(f'7 parameter fitting is not enough, try 10 parameter fitting...')
                self.ctr2_iflux, self.ctr2_lon, self.ctr2_lat, self.ctr3_iflux, self.ctr3_lon, self.ctr3_lat = self.find_first_second_nearby_ps_lon_lat()
                self.n_pts = 0
                self.n_call = 0
                obj_minuit = Minuit(lsq_10_params, norm_beam1=self.ini_norm_beam, ctr1_lon_shift=0, ctr1_lat_shift=0, norm_beam2=self.ctr2_iflux, ctr2_lon_shift=0, ctr2_lat_shift=0, norm_beam3=self.ctr3_iflux, ctr3_lon_shift=0, ctr3_lat_shift=0, const=0)

                obj_minuit.limits = [(0,1),(-0.4,0.4),(-0.4,0.4),(0,1),(-0.4,0.4),(-0.4,0.4),(0,1),(-0.5,0.5),(-0.5,0.5), (-100,100)]
                print(obj_minuit.migrad())
                print(obj_minuit.hesse())

                ndof = self.n_pts / (self.n_call - 10)
                chi2dof = obj_minuit.fval / ndof
                str_chi2 = f"𝜒²/ndof = {obj_minuit.fval:.2f} / {ndof} = {chi2dof}"
                print(str_chi2)

                is_ps = obj_minuit.values['norm_beam1'] > self.sigma_threshold * obj_minuit.errors['norm_beam1']

                if obj_minuit.fmin.hesse_failed:
                    raise ValueError('hesse failed!')
                elif (chi2dof < 1.06):
                    if is_ps:
                        print(f'10 parameter fitting is enough, hesse ok')
                        fit_lon = self.lon + obj_minuit.values['ctr1_lon_shift']
                        fit_lat = self.lon + obj_minuit.values['ctr1_lat_shift']
                        self.fit_norm_beam = obj_minuit.values['norm_beam1']
                        true_norm_beam = self.flux2norm_beam(self.iflux)
                        print(f'{true_norm_beam=}')
                        print(f'fit_norm_beam={self.fit_norm_beam}')
                        return obj_minuit.values['norm_beam1'], fit_lon, fit_lat
                    else:
                        print(f'not a point source under {self.sigma_threshold} sigma threshold')
                        return 0.0, self.lon, self.lat
                else:
                    self.do_13_params = True
            if self.do_13_params:
                print(f'10 parameter fitting is not enough, try 13 parameter fitting...')
                self.ctr2_iflux, self.ctr2_lon, self.ctr2_lat, self.ctr3_iflux, self.ctr3_lon, self.ctr3_lat, self.ctr4_iflux, self.ctr4_lon, self.ctr4_lat = self.find_first_second_third_nearby_ps_lon_lat()
                self.n_pts = 0
                self.n_call = 0
                obj_minuit = Minuit(lsq_13_params, norm_beam1=self.ini_norm_beam, ctr1_lon_shift=0, ctr1_lat_shift=0, norm_beam2=self.ctr2_iflux, ctr2_lon_shift=0, ctr2_lat_shift=0, norm_beam3=self.ctr3_iflux, ctr3_lon_shift=0, ctr3_lat_shift=0, norm_beam4=self.ctr4_iflux, ctr4_lon_shift=0, ctr4_lat_shift=0, const=0)

                obj_minuit.limits = [(0,1),(-0.3,0.3),(-0.3,0.3),(0,1),(-0.4,0.4),(-0.4,0.4),(0,1),(-0.5,0.5),(-0.5,0.5), (0,1),(-0.5,0.5),(-0.5,0.5), (-100,100)]
                print(obj_minuit.migrad())
                print(obj_minuit.hesse())
                ndof = self.n_pts / (self.n_call - 13)
                chi2dof = obj_minuit.fval / ndof
                str_chi2 = f"𝜒²/ndof = {obj_minuit.fval:.2f} / {ndof} = {chi2dof}"
                print(str_chi2)

                is_ps = obj_minuit.values['norm_beam1'] > self.sigma_threshold * obj_minuit.errors['norm_beam1']

                if obj_minuit.fmin.hesse_failed:
                    raise ValueError('hesse failed!')
                elif (chi2dof < 1.06):
                    if is_ps:
                        print(f'13 parameter fitting is enough, hesse ok')
                        fit_lon = self.lon + obj_minuit.values['ctr1_lon_shift']
                        fit_lat = self.lon + obj_minuit.values['ctr1_lat_shift']
                        self.fit_norm_beam = obj_minuit.values['norm_beam1']
                        true_norm_beam = self.flux2norm_beam(self.iflux)
                        print(f'{true_norm_beam=}')
                        print(f'fit_norm_beam={self.fit_norm_beam}')
                        return obj_minuit.values['norm_beam1'], fit_lon, fit_lat
                    else:
                        print(f'not a point source under {self.sigma_threshold} sigma threshold')
                        return 0.0, self.lon, self.lat
                else:
                    self.do_16_params = True

            if self.do_16_params:
                print(f'13 parameter fitting is not enough, try 16 parameter fitting...')
                self.ctr2_iflux, self.ctr2_lon, self.ctr2_lat, self.ctr3_iflux, self.ctr3_lon, self.ctr3_lat, self.ctr4_iflux, self.ctr4_lon, self.ctr4_lat, self.ctr5_iflux, self.ctr5_lon, self.ctr5_lat= self.find_first_second_third_fourth_nearby_ps_lon_lat()
                self.n_pts = 0
                self.n_call = 0
                obj_minuit = Minuit(lsq_16_params, norm_beam1=self.ini_norm_beam, ctr1_lon_shift=0, ctr1_lat_shift=0, norm_beam2=self.ctr2_iflux, ctr2_lon_shift=0, ctr2_lat_shift=0, norm_beam3=self.ctr3_iflux, ctr3_lon_shift=0, ctr3_lat_shift=0, norm_beam4=self.ctr4_iflux, ctr4_lon_shift=0, ctr4_lat_shift=0, norm_beam5=self.ctr5_iflux, ctr5_lon_shift=0, ctr5_lat_shift=0, const=0)

                obj_minuit.limits = [(0,1),(-0.3,0.3),(-0.3,0.3),(0,1),(-0.3,0.3),(-0.3,0.3),(0,1),(-0.5,0.5),(-0.5,0.5), (0,1),(-0.3,0.3),(-0.3,0.3),(0,1),(-0.5,0.5),(-0.5,0.5),(-100,100)]
                print(obj_minuit.migrad())
                print(obj_minuit.hesse())
                ndof = self.n_pts / (self.n_call - 16)
                chi2dof = obj_minuit.fval / ndof
                str_chi2 = f"𝜒²/ndof = {obj_minuit.fval:.2f} / {ndof} = {chi2dof}"
                print(str_chi2)

                is_ps = obj_minuit.values['norm_beam1'] > self.sigma_threshold * obj_minuit.errors['norm_beam1']

                if obj_minuit.fmin.hesse_failed:
                    raise ValueError('hesse failed!')
                elif (chi2dof < 1.06):
                    if is_ps:
                        print(f'16 parameter fitting is enough, hesse ok')
                        fit_lon = self.lon + obj_minuit.values['ctr1_lon_shift']
                        fit_lat = self.lon + obj_minuit.values['ctr1_lat_shift']
                        self.fit_norm_beam = obj_minuit.values['norm_beam1']
                        true_norm_beam = self.flux2norm_beam(self.iflux)
                        print(f'{true_norm_beam=}')
                        print(f'fit_norm_beam={self.fit_norm_beam}')
                        return obj_minuit.values['norm_beam1'], fit_lon, fit_lat
                    else:
                        print(f'not a point source under {self.sigma_threshold} sigma threshold')
                        return 0.0, self.lon, self.lat
                else:
                    raise ValueError('16 parameter fitting is not enough, try something else!')

        if mode == '2params':
            obj_minuit = Minuit(lsq_2_params, norm_beam=0.5, const=0)
            # obj_minuit = Minuit(lsq1, norm_beam=1, const=0)
            obj_minuit.limits = [(0,10), (-100,100)]
            # obj_minuit.limits = [(0,10), (-100,100)]
            print(obj_minuit.migrad())
            print(obj_minuit.hesse())
            ndof = self.n_pts / (self.n_call - 2)
            chi2dof = obj_minuit.fval / ndof
            str_chi2 = f"𝜒²/ndof = {obj_minuit.fval:.2f} / {ndof} = {chi2dof}"
            print(str_chi2)

        if mode == '4params':
            obj_minuit = Minuit(lsq_4_params, norm_beam=0.3, ctr_lon_shift=0, ctr_lat_shift=0, const=0)
            # obj_minuit = Minuit(lsq1, norm_beam=1, const=0)
            obj_minuit.limits = [(0,1), (-0.1,0.1), (-0.1,0.1), (-100,100)]
            # obj_minuit.limits = [(0,10), (-100,100)]
            print(obj_minuit.migrad())
            print(obj_minuit.hesse())
            ndof = self.n_pts / (self.n_call - 4)
            chi2dof = obj_minuit.fval / ndof
            str_chi2 = f"𝜒²/ndof = {obj_minuit.fval:.2f} / {ndof} = {chi2dof}"
            print(str_chi2)

        if mode == '7params':
            obj_minuit = Minuit(lsq_7_params, norm_beam1=0.5, ctr1_lon_shift=0, ctr1_lat_shift=0, norm_beam2=0.5, ctr2_lon_shift=0, ctr2_lat_shift=0, const=0)
            self.ctr2_lon, self.ctr2_lat = self.lon, self.lat
            obj_minuit.limits = [(0,10), (-1,1), (-1,1),(0,10),(-3,3),(-3,3), (-100,100)]
            print(obj_minuit.migrad())
            print(obj_minuit.hesse())
            ndof = self.n_pts / (self.n_call - 7)
            chi2dof = obj_minuit.fval / ndof
            str_chi2 = f"𝜒²/ndof = {obj_minuit.fval:.2f} / {ndof} = {chi2dof}"
            print(str_chi2)

        if mode == '10params':
            self.ctr2_lon, self.ctr2_lat, self.ctr3_lon, self.ctr3_lat = self.find_first_second_nearby_ps_lon_lat()
            self.n_pts = 0
            self.n_call = 0
            obj_minuit = Minuit(lsq_10_params, norm_beam1=0.068, ctr1_lon_shift=0, ctr1_lat_shift=0, norm_beam2=0.184, ctr2_lon_shift=0, ctr2_lat_shift=0, norm_beam3=0.0, ctr3_lon_shift=0, ctr3_lat_shift=0, const=0)

            obj_minuit.limits = [(0,1),(-0.1,0.1),(-0.1,0.1),(0,1),(-0.2,0.2),(-0.2,0.2),(0,1),(-0.3,0.3),(-0.3,0.3), (-100,100)]
            print(obj_minuit.migrad())
            print(obj_minuit.hesse())
            ndof = self.n_pts / (self.n_call - 10)
            chi2dof = obj_minuit.fval / ndof
            str_chi2 = f"𝜒²/ndof = {obj_minuit.fval:.2f} / {ndof} = {chi2dof}"
            print(str_chi2)



if __name__ == '__main__':
    m = np.load('../../FGSim/PSNOISE/2048/40.npy')[0]
    nstd = np.load('../../FGSim/NSTDNORTH/2048/40.npy')[0]
    df_mask = pd.read_csv('../partial_sky_ps/ps_in_mask/mask40.csv')
    flux_idx = 12
    lon = np.rad2deg(df_mask.at[flux_idx, 'lon'])
    lat = np.rad2deg(df_mask.at[flux_idx, 'lat'])
    iflux = df_mask.at[flux_idx, 'iflux']

    df_ps = pd.read_csv('../../test/ps_sort/sort_by_iflux/40.csv')
    
    lmax = 350
    nside = 2048
    beam = 63
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax)
    cl_cmb = np.load('../../src/cmbsim/cmbdata/cmbcl.npy')[:lmax+1,0]
    cl_cmb = cl_cmb * bl**2

    obj = FitPointSource(m=m, nstd=nstd, flux_idx=flux_idx, df_mask=df_mask, df_ps=df_ps, cl_cmb=cl_cmb, lon=lon, lat=lat, iflux=iflux, lmax=lmax, nside=nside, radius_factor=1.0, beam=beam)

    obj.see_true_map(m=m, lon=lon, lat=lat, nside=nside, beam=beam)
    # obj.find_nearby_ps_lon_lat()
    # obj.find_first_second_nearby_ps_lon_lat()
    # obj.fit_ps_ns(mode='10params')
    obj.fit_ps_ns()

