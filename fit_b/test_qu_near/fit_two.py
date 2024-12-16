import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pandas as pd
import time
import pickle
import os,sys
import logging
import ipdb

from pathlib import Path
from iminuit import Minuit
from iminuit.cost import LeastSquares
from numpy.polynomial.legendre import Legendre
from scipy.interpolate import CubicSpline
from memory_profiler import profile

# logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s -%(name)s - %(message)s')
logging.basicConfig(level=logging.WARNING)
# logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.INFO)

class FitPointSource:
    def __init__(self, m, nstd, nside, flux_idx, radius_factor, beam, distance_factor, sigma_threshold=5, epsilon=1e-4, debug_flag=False):
        self.m = m # sky maps (npix,)
        self.nstd = nstd # noise standard deviation
        self.nside = nside # resolution of healpy maps
        self.flux_idx = flux_idx
        self.radius_factor = radius_factor # disc radius of fitting region
        self.sigma_threshold = sigma_threshold # judge if a signal is a point source
        self.beam = beam # arcmin
        self.distance_factor = distance_factor
        self.epsilon = epsilon # if CMB covariance matrix is not semi-positive, add this to cross term

        self.sigma = np.deg2rad(beam) / 60 / (np.sqrt(8 * np.log(2)))
        self.lon = np.load('./data/ps/pix_lon1.npy')
        self.lat = np.load('./data/ps/pix_lat1.npy')

        lon_2 = np.load(f'./data/ps/pix_lon2_{distance_factor}.npy')
        lat_2 = np.load(f'./data/ps/pix_lat2_{distance_factor}.npy')

        ctr0_pix = hp.ang2pix(nside=self.nside, theta=self.lon, phi=self.lat, lonlat=True)
        ctr1_pix = hp.ang2pix(nside=self.nside, theta=lon_2, phi=lat_2, lonlat=True)
        self.ctr0_vec = np.array(hp.pix2vec(nside=self.nside, ipix=ctr0_pix)).astype(np.float64)
        self.ctr1_vec = np.array(hp.pix2vec(nside=self.nside, ipix=ctr1_pix)).astype(np.float64)

        ipix_fit_0 = hp.query_disc(nside=self.nside, vec=self.ctr0_vec, radius=self.radius_factor * np.deg2rad(self.beam) / 60)
        ipix_fit_1 = hp.query_disc(nside=self.nside, vec=self.ctr1_vec, radius=self.radius_factor * np.deg2rad(self.beam) / 60)


        self.ipix_fit = np.union1d(ar1=ipix_fit_0, ar2=ipix_fit_1)

        path_pix_idx = Path('./pix_idx')
        path_pix_idx.mkdir(exist_ok=True, parents=True)
        np.save(path_pix_idx / Path(f'{flux_idx}.npy'), self.ipix_fit)

        self.vec_around = np.array(hp.pix2vec(nside=self.nside, ipix=self.ipix_fit.astype(int))).astype(np.float64)
        self.ndof = len(self.ipix_fit)

        self.num_near_ps = 0
        self.flag_too_near = False

        self.nside2pixarea = hp.nside2pixarea(nside=nside)
        logger.info(f'{beam=}, {radius_factor=}, {self.lon=}, {self.lat=}, ndof={self.ndof}')

    def calc_covariance_matrix(self, mode='cmb+noise'):

        if mode == 'noise':
            nstd2 = self.nstd**2
            # cov = np.zeros((self.ndof,self.ndof))
            cov = nstd2 * np.eye(self.ndof, self.ndof)

            # for i in range(self.ndof):
            #     cov[i,i] = cov[i,i] + nstd2

            logger.debug(f'{cov=}')
            self.inv_cov = np.linalg.inv(cov)
            path_inv_cov = Path(f'inv_cov_{self.nside}/r_{self.radius_factor}') / Path(mode)
            path_inv_cov.mkdir(parents=True, exist_ok=True)
            np.save(path_inv_cov / Path(f'{self.flux_idx}.npy'), self.inv_cov)
            return None

    def input_lonlat2pix_lonlat(self, input_lon, input_lat):
        ipix = hp.ang2pix(nside=self.nside, theta=input_lon, phi=input_lat, lonlat=True)
        out_lon, out_lat = hp.pix2ang(nside=self.nside, ipix=ipix, lonlat=True)
        return out_lon, out_lat

    def adjust_lat(self, lat):
        if lat < -90 or lat > 90:
            lat = lat % 360
            if lat < -90:
                lat = -180 - lat
            if (lat > 90) and (lat <= 270):
                lat = 180 - lat
            elif lat > 270:
                lat = lat - 360
        return lat

    def find_nearby_ps(self, num_ps=1):
        threshold_factor = self.radius_factor + 1.1
        logger.debug(f'{threshold_factor=}')
        dir_0 = (self.lon, self.lat)
        arr_1 = self.df_ps.loc[:, 'flux_idx']
        logger.debug(f'{arr_1.shape=}')
        bool_arr = self.df_ps.loc[:, 'flux_idx'] != self.df_mask.at[self.flux_idx, 'flux_idx']
        logger.debug(f'{bool_arr.shape=}')
        lon_other = np.rad2deg(self.df_ps.loc[bool_arr, 'lon'])
        lat_other = np.rad2deg(self.df_ps.loc[bool_arr, 'lat'])
        dir_other = (lon_other, lat_other)
        ang = np.rad2deg(hp.rotator.angdist(dir1=dir_0, dir2=dir_other, lonlat=True))
        logger.debug(f'{ang.shape=}')
        threshold = threshold_factor * self.beam / 60
        logger.debug(f'{ang=}')

        if len(ang) - np.count_nonzero(ang) > 0:
            logger.debug(f'there are some point sources overlap')
            index_near = np.nonzero(np.where(ang==0, 1, 0))
        else:
            index_near = np.nonzero(np.where((ang < threshold), ang, 0))

        ang_near = ang[index_near]
        logger.debug(f'{index_near=}')
        logger.debug(f'{ang_near=}')

        logger.debug(f'{index_near[0].shape=}')
        logger.debug(f'{ang_near.shape=}')

        logger.debug(f'number of ir, radio ps = {index_near[0].size}')

        lon_list = []
        lat_list = []
        iflux_list = []
        for i in range(min(num_ps, len(index_near[0]))):
            index = index_near[0][i]
            if index < self.df_mask.at[self.flux_idx, 'rank']:
                lon = np.rad2deg(self.df_ps.at[index, 'lon'])
                lat = np.rad2deg(self.df_ps.at[index, 'lat'])
                iflux = self.flux2norm_beam(self.df_ps.at[index, 'iflux'])
            else:
                lon = np.rad2deg(self.df_ps.at[index + 1, 'lon'])
                lat = np.rad2deg(self.df_ps.at[index + 1, 'lat'])
                iflux = self.flux2norm_beam(self.df_ps.at[index + 1, 'iflux'])
            lon_list.append(lon)
            lat_list.append(lat)
            iflux_list.append(iflux)

        logger.debug(f'{iflux_list=}')
    
        ##Optional visualization code commented out
        #hp.gnomview(self.m, rot=[self.lon,self.lat,0])
        #for lon, lat in zip(lon_list, lat_list):
        #    hp.projscatter(lon, lat, lonlat=True)
        #plt.show()
    
        # return tuple(iflux_list + lon_list + lat_list)
        iflux_arr = np.array(iflux_list)
        ang_near_arr = np.array(ang_near)[0:num_ps]
        lon_arr = np.array(lon_list)
        lat_arr = np.array(lat_list)
        num_ps = np.count_nonzero(np.where(iflux_arr > self.flux2norm_beam(flux=1), iflux_arr, 0))
        logger.debug(f'there are {num_ps} ps > 1 mJy')
        logger.debug(f'ang_near_arr before mask very faint: {ang_near_arr}')
        logger.debug(f'lon_arr before mask very faint: {lon_arr}')
        logger.debug(f'lat_arr before mask very faint: {lat_arr}')
        logger.debug(f'iflux_arr before mask very faint: {iflux_arr}')

        mask_very_faint = iflux_arr > self.flux2norm_beam(flux=1)

        ang_near_arr = ang_near_arr[mask_very_faint].copy()
        iflux_arr = iflux_arr[mask_very_faint].copy()
        lon_arr = lon_arr[mask_very_faint].copy()
        lat_arr = lat_arr[mask_very_faint].copy()

        self.ang_near = ang_near_arr

        logger.debug(f'ang_near_arr after mask very faint: {ang_near_arr}')
        logger.debug(f'lon_arr after mask very faint: {lon_arr}')
        logger.debug(f'lat_arr after mask very faint: {lat_arr}')
        logger.debug(f'iflux_arr after mask very faint: {iflux_arr}')

        if num_ps > 0:
            ang_near_and_bigger_than_threshold = ang_near[0:num_ps]
            if any(ang_near_and_bigger_than_threshold < 0.35):
                self.flag_too_near = True

                self.num_near_ps = np.count_nonzero(np.where(ang_near_and_bigger_than_threshold < 0.35, ang_near_and_bigger_than_threshold, 0))
                logger.debug(f'{self.num_near_ps=}')
                sorted_indices = np.argsort(ang_near_arr)

                ang_near_arr = ang_near_arr[sorted_indices]
                iflux_arr = iflux_arr[sorted_indices]
                lon_arr = lon_arr[sorted_indices]
                lat_arr = lat_arr[sorted_indices]

                logger.debug(f'ang_near_arr after sort by ang: {ang_near_arr}')
                logger.debug(f'lon_arr after sort by ang: {lon_arr}')
                logger.debug(f'lat_arr after sort by ang: {lat_arr}')
                logger.debug(f'iflux_arr after sort by ang: {iflux_arr}')

            logger.debug(f'{self.flag_too_near = }')

        return num_ps, tuple(sum(zip(iflux_arr, lon_arr, lat_arr), ()))

    def fit_all(self, cov_mode:str, mode:str='pipeline'):
        def calc_error():
            theta = hp.rotator.angdist(dir1=ctr0_vec, dir2=vec_around)

            def model():
                return 1 / (2 * np.pi * self.sigma**2) * np.exp(- (theta)**2 / (2 * self.sigma**2))

            y_model = model()
            Fish_mat = y_model @ self.inv_cov @ y_model
            sigma = 1 / np.sqrt(Fish_mat)
            logger.info(f'{sigma=}')

        def lsq_2_params(norm_beam, const):

            theta = hp.rotator.angdist(dir1=ctr0_vec, dir2=vec_around)

            def model():
                return norm_beam / (2 * np.pi * self.sigma**2) * np.exp(- (theta)**2 / (2 * self.sigma**2)) + const
                # return norm_beam / (2 * np.pi * self.sigma**2) * np.exp(- (theta)**2 / (2 * self.sigma**2))
                # return 1 / (2 * np.pi * self.sigma**2) * np.exp(- (theta)**2 / (2 * self.sigma**2))

            y_model = model()
            y_data = self.m[ipix_fit]
            y_err = self.nstd[ipix_fit]
            y_diff = y_data - y_model

            # error_estimate = np.sum(y_model**2 / y_err**2)
            # print(f"{error_estimate=}")

            z = (y_diff) @ self.inv_cov @ (y_diff)
            return z

        def lsq_params(*args):
            # args is expected to be in the format:
            # norm_beam1, norm_beamN, const
        
            num_ps = (len(args))  # Determine the number of point sources based on the number of arguments
        
            # Process each point source
            thetas = []
            for i in range(num_ps):
                norm_beam = args[i]
                lon = self.fit_lon[i]
                lat = self.fit_lat[i]
                if np.isnan(lon): lon = self.fit_lon[i]+np.random.uniform(-0.01, 0.01)
                if np.isnan(lat): lat = self.fit_lat[i]+np.random.uniform(-0.01, 0.01)
                # print(f'{lon=},{lat=}')
                lat = self.adjust_lat(lat)
                ctr_vec = np.array(hp.ang2vec(theta=lon, phi=lat, lonlat=True))
        
                theta = hp.rotator.angdist(dir1=ctr_vec, dir2=vec_around)
                thetas.append(self.nside2pixarea * norm_beam / (2 * np.pi * self.sigma**2) * np.exp(- (theta)**2 / (2 * self.sigma**2)))
        
            def model():
                return sum(thetas)
        
            y_model = model()
            y_data = self.m[ipix_fit]
            y_err = self.nstd * np.ones_like(y_data)
            # y_err = self.nstd[ipix_fit]
        
            y_diff = y_data - y_model

            z = (y_diff) @ self.inv_cov @ (y_diff)
            logger.debug(f'{z=}')
            return z

        self.inv_cov = np.load(f'./inv_cov_{self.nside}/r_{self.radius_factor}/{cov_mode}/{self.flux_idx}.npy')

        ctr0_vec = self.ctr0_vec
        ipix_fit = self.ipix_fit
        vec_around = self.vec_around

        logger.info(f'Ready for fitting...')

        def fit_2_params():
            params = (5e2,)
            self.fit_lon = (self.lon,)
            self.fit_lat = (self.lat,)
            logger.debug(f'{self.fit_lon=}, {self.fit_lat=}')

            obj_minuit = Minuit(lsq_params, name=("norm_beam1",), *params)
            obj_minuit.limits = [(-1000,1000),]
            logger.debug(f'\n{obj_minuit.migrad()}')
            logger.debug(f'\n{obj_minuit.hesse()}')

            chi2dof = obj_minuit.fval / self.ndof
            str_chi2 = f"𝜒²/ndof = {obj_minuit.fval:.2f} / {self.ndof} = {chi2dof}"
            logger.debug(str_chi2)

            if obj_minuit.fmin.hesse_failed:
                raise ValueError('hesse failed!')

            logger.info(f'2 parameter fitting is enough, hesse ok')
            return chi2dof, obj_minuit.values['norm_beam1'],obj_minuit.errors['norm_beam1']

        def fit_3_params():
            # num_ps, (self.ctr2_iflux, self.ctr2_lon, self.ctr2_lat) = self.find_nearby_ps(num_ps=1)
            self.ctr2_iflux = 3e2
            self.ctr2_lon = np.load(f'./data/ps/pix_lon2_{self.distance_factor}.npy')
            self.ctr2_lat = np.load(f'./data/ps/pix_lat2_{self.distance_factor}.npy')
            
            params = (5e2, self.ctr2_iflux)
            self.fit_lon = (self.lon, self.ctr2_lon)
            self.fit_lat = (self.lat, self.ctr2_lat)
            obj_minuit = Minuit(lsq_params, name=("norm_beam1","norm_beam2"), *params)

            obj_minuit.limits = [(-1000,1000),(-1000,1000)]
            logger.debug(f'\n{obj_minuit.migrad()}')
            logger.debug(f'\n{obj_minuit.hesse()}')

            chi2dof = obj_minuit.fval / self.ndof
            str_chi2 = f"𝜒²/ndof = {obj_minuit.fval:.2f} / {self.ndof} = {chi2dof}"
            logger.debug(str_chi2)

            if obj_minuit.fmin.hesse_failed:
                raise ValueError('hesse failed!')

            logger.info(f'3 parameter fitting is enough, hesse ok')
            return chi2dof, obj_minuit.values['norm_beam1'],obj_minuit.errors['norm_beam1']

        def fit_4_params():
            num_ps, (self.ctr2_iflux, self.ctr2_lon, self.ctr2_lat, self.ctr3_iflux, self.ctr3_lon, self.ctr3_lat) = self.find_nearby_ps(num_ps=2)

            params = (self.ini_norm_beam, self.ctr2_iflux, self.ctr3_iflux, 0)
            self.fit_lon = (self.lon, self.ctr2_lon, self.ctr3_lon)
            self.fit_lat = (self.lat, self.ctr2_lat, self.ctr3_lat)
            obj_minuit = Minuit(lsq_params, name=("norm_beam1","norm_beam2","norm_beam3","const"), *params)


            obj_minuit.limits = [(-1,1),(-1,1),(-1,1),(-1000,1000)]
            logger.debug(f'\n{obj_minuit.migrad()}')
            logger.debug(f'\n{obj_minuit.hesse()}')

            chi2dof = obj_minuit.fval / self.ndof
            str_chi2 = f"𝜒²/ndof = {obj_minuit.fval:.2f} / {self.ndof} = {chi2dof}"
            logger.debug(str_chi2)

            if obj_minuit.fmin.hesse_failed:
                raise ValueError('hesse failed!')

            logger.info(f'4 parameter fitting is enough, hesse ok')
            return chi2dof, obj_minuit.values['norm_beam1'],obj_minuit.errors['norm_beam1']

        def fit_5_params():
            num_ps, (self.ctr2_iflux, self.ctr2_lon, self.ctr2_lat, self.ctr3_iflux, self.ctr3_lon, self.ctr3_lat, self.ctr4_iflux, self.ctr4_lon, self.ctr4_lat) = self.find_nearby_ps(num_ps=3)
            params = (self.ini_norm_beam, self.ctr2_iflux,self.ctr3_iflux, self.ctr4_iflux, 0)

            self.fit_lon = (self.lon, self.ctr2_lon, self.ctr3_lon, self.ctr4_lon)
            self.fit_lat = (self.lat, self.ctr2_lat, self.ctr3_lat, self.ctr4_lat)
            obj_minuit = Minuit(lsq_params, name=("norm_beam1","norm_beam2","norm_beam3","norm_beam4","const"), *params)


            obj_minuit.limits = [(-1,1),(-1,1),(-1,1),(-1,1), (-1000,1000)]
            logger.debug(f'\n{obj_minuit.migrad()}')
            logger.debug(f'\n{obj_minuit.hesse()}')

            chi2dof = obj_minuit.fval / self.ndof
            str_chi2 = f"𝜒²/ndof = {obj_minuit.fval:.2f} / {self.ndof} = {chi2dof}"
            logger.debug(str_chi2)

            if obj_minuit.fmin.hesse_failed:
                raise ValueError('hesse failed!')

            logger.info(f'5 parameter fitting is enough, hesse ok')

            return chi2dof, obj_minuit.values['norm_beam1'],obj_minuit.errors['norm_beam1']

        def fit_6_params():
            num_ps, (self.ctr2_iflux, self.ctr2_lon, self.ctr2_lat, self.ctr3_iflux, self.ctr3_lon, self.ctr3_lat, self.ctr4_iflux, self.ctr4_lon, self.ctr4_lat, self.ctr5_iflux, self.ctr5_lon, self.ctr5_lat) = self.find_nearby_ps(num_ps=4)

            params = (self.ini_norm_beam, self.ctr2_iflux, self.ctr3_iflux, self.ctr4_iflux, self.ctr5_iflux,0)
            self.fit_lon = (self.lon, self.ctr2_lon, self.ctr3_lon, self.ctr4_lon, self.ctr5_lon)
            self.fit_lat = (self.lat, self.ctr2_lat, self.ctr3_lat, self.ctr4_lat, self.ctr5_lat)
            obj_minuit = Minuit(lsq_params, name=("norm_beam1","norm_beam2","norm_beam3","norm_beam4","norm_beam5","const"), *params)

            obj_minuit.limits = [(-1,1),(-1,1),(-1,1), (-1,1),(-1,1), (-1000,1000)]
            logger.debug(f'\n{obj_minuit.migrad()}')
            logger.debug(f'\n{obj_minuit.hesse()}')

            chi2dof = obj_minuit.fval / self.ndof
            str_chi2 = f"𝜒²/ndof = {obj_minuit.fval:.2f} / {self.ndof} = {chi2dof}"
            logger.debug(str_chi2)

            if obj_minuit.fmin.hesse_failed:
                raise ValueError('hesse failed!')

            logger.info(f'6 parameter fitting is enough, hesse ok')

            return chi2dof, obj_minuit.values['norm_beam1'], obj_minuit.errors['norm_beam1']

        def fit_7_params():
            num_ps, (self.ctr2_iflux, self.ctr2_lon, self.ctr2_lat, self.ctr3_iflux, self.ctr3_lon, self.ctr3_lat, self.ctr4_iflux, self.ctr4_lon, self.ctr4_lat, self.ctr5_iflux, self.ctr5_lon, self.ctr5_lat, self.ctr6_iflux, self.ctr6_lon, self.ctr6_lat) = self.find_nearby_ps(num_ps=5)

            params = (self.ini_norm_beam, self.ctr2_iflux, self.ctr3_iflux, self.ctr4_iflux, self.ctr5_iflux, self.ctr6_iflux, 0)
            self.fit_lon = (self.lon, self.ctr2_lon, self.ctr3_lon, self.ctr4_lon, self.ctr5_lon, self.ctr6_lon)
            self.fit_lat = (self.lat, self.ctr2_lat, self.ctr3_lat, self.ctr4_lat, self.ctr5_lat, self.ctr6_lat)
            obj_minuit = Minuit(lsq_params, name=("norm_beam1","ctr1_lon_shift","ctr1_lat_shift","norm_beam2","ctr2_lon_shift","ctr2_lat_shift","norm_beam3","ctr3_lon_shift","ctr3_lat_shift","norm_beam4","ctr4_lon_shift","ctr4_lat_shift","norm_beam5","ctr5_lon_shift","ctr5_lat_shift","norm_beam6","ctr6_lon_shift","ctr6_lat_shift","const"), *params)

            obj_minuit.limits = [(-1,1),(-1,1),(-1,1),(-1,1),(-1,1),(-1,1),(-1000,1000)]
            logger.debug(f'\n{obj_minuit.migrad()}')
            logger.debug(f'\n{obj_minuit.hesse()}')

            chi2dof = obj_minuit.fval / self.ndof
            str_chi2 = f"𝜒²/ndof = {obj_minuit.fval:.2f} / {self.ndof} = {chi2dof}"
            logger.debug(str_chi2)

            if obj_minuit.fmin.hesse_failed:
                raise ValueError('hesse failed!')

            logger.info(f'7 parameter fitting is enough, hesse ok')

            return chi2dof, obj_minuit.values['norm_beam1'], obj_minuit.errors['norm_beam1']

        def fit_8_params():
            num_ps, (self.ctr2_iflux, self.ctr2_lon, self.ctr2_lat, self.ctr3_iflux, self.ctr3_lon, self.ctr3_lat, self.ctr4_iflux, self.ctr4_lon, self.ctr4_lat, self.ctr5_iflux, self.ctr5_lon, self.ctr5_lat, self.ctr6_iflux, self.ctr6_lon, self.ctr6_lat, self.ctr7_iflux, self.ctr7_lon, self.ctr7_lat) = self.find_nearby_ps(num_ps=6)

            params = (self.ini_norm_beam, self.ctr2_iflux, self.ctr3_iflux, self.ctr4_iflux, self.ctr5_iflux, self.ctr6_iflux, self.ctr7_iflux, 0)
            self.fit_lon = (self.lon, self.ctr2_lon, self.ctr3_lon, self.ctr4_lon, self.ctr5_lon, self.ctr6_lon, self.ctr7_lon)
            self.fit_lat = (self.lat, self.ctr2_lat, self.ctr3_lat, self.ctr4_lat, self.ctr5_lat, self.ctr6_lat, self.ctr7_lat)
            obj_minuit = Minuit(lsq_params, name=("norm_beam1","norm_beam2","norm_beam3","norm_beam4","norm_beam5","norm_beam6","norm_beam7","const"), *params)

            obj_minuit.limits = [(-1,1),(-1,1),(-1,1), (-1,1),(-1,1),(-1,1),(-1,1), (-1000,1000)]
            logger.debug(f'\n{obj_minuit.migrad()}')
            logger.debug(f'\n{obj_minuit.hesse()}')

            chi2dof = obj_minuit.fval / self.ndof
            str_chi2 = f"𝜒²/ndof = {obj_minuit.fval:.2f} / {self.ndof} = {chi2dof}"
            logger.debug(str_chi2)

            if obj_minuit.fmin.hesse_failed:
                raise ValueError('hesse failed!')

            logger.info(f'8 parameter fitting is enough, hesse ok')

            return chi2dof, obj_minuit.values['norm_beam1'], obj_minuit.errors['norm_beam1']

        def fit_9_params():
            num_ps, (self.ctr2_iflux, self.ctr2_lon, self.ctr2_lat, self.ctr3_iflux, self.ctr3_lon, self.ctr3_lat, self.ctr4_iflux, self.ctr4_lon, self.ctr4_lat, self.ctr5_iflux, self.ctr5_lon, self.ctr5_lat, self.ctr6_iflux, self.ctr6_lon, self.ctr6_lat, self.ctr7_iflux, self.ctr7_lon, self.ctr7_lat, self.ctr8_iflux, self.ctr8_lon, self.ctr8_lat) = self.find_nearby_ps(num_ps=7)

            params = (self.ini_norm_beam, self.ctr2_iflux, self.ctr3_iflux, self.ctr4_iflux, self.ctr5_iflux, self.ctr6_iflux, self.ctr7_iflux, self.ctr8_iflux, 0)
            self.fit_lon = (self.lon, self.ctr2_lon, self.ctr3_lon, self.ctr4_lon, self.ctr5_lon, self.ctr6_lon, self.ctr7_lon, self.ctr8_lon)
            self.fit_lat = (self.lat, self.ctr2_lat, self.ctr3_lat, self.ctr4_lat, self.ctr5_lat, self.ctr6_lat, self.ctr7_lat, self.ctr8_lon)
            obj_minuit = Minuit(lsq_params, name=("norm_beam1","norm_beam2","norm_beam3","norm_beam4","norm_beam5","norm_beam6","norm_beam7","norm_beam8","const"), *params)


            obj_minuit.limits = [(-1,1),(-1,1),(-1,1),(-1,1),(-1,1),(-1,1),(-1,1),(-1,1), (-1000,1000)]
            logger.debug(f'\n{obj_minuit.migrad()}')
            logger.debug(f'\n{obj_minuit.hesse()}')

            chi2dof = obj_minuit.fval / self.ndof
            str_chi2 = f"𝜒²/ndof = {obj_minuit.fval:.2f} / {self.ndof} = {chi2dof}"
            logger.debug(str_chi2)

            if obj_minuit.fmin.hesse_failed:
                raise ValueError('hesse failed!')

            logger.info(f'9 parameter fitting is enough, hesse ok')

            return chi2dof, obj_minuit.values['norm_beam1'], obj_minuit.errors['norm_beam1']

        logger.info(f'begin point source fitting, first do 2 parameter fit...')
        chi2dof, fit_norm, norm_error = fit_2_params()
        if fit_norm < self.sigma_threshold * norm_error:
            logger.info('there is no point sources.')

        num_ps = 1
        true_norm_beam = 5e2

        if mode == 'pipeline':

            if num_ps == 0:
                chi2dof, fit_norm, norm_error = fit_2_params()
            elif num_ps == 1:
                chi2dof, fit_norm, norm_error = fit_3_params()
            elif num_ps == 2:
                chi2dof, fit_norm, norm_error = fit_4_params()
            elif num_ps == 3:
                chi2dof, fit_norm, norm_error = fit_5_params()
            elif num_ps == 4:
                chi2dof, fit_norm, norm_error = fit_6_params()
            elif num_ps == 5:
                chi2dof, fit_norm, norm_error = fit_7_params()
            elif num_ps == 6:
                chi2dof, fit_norm, norm_error = fit_8_params()
            elif num_ps == 7:
                chi2dof, fit_norm, norm_error = fit_9_params()

            fit_error = np.abs(fit_norm - true_norm_beam) / true_norm_beam

            logger.info(f'{num_ps=}, {chi2dof=}, {fit_norm=}, {norm_error=}')
            logger.info(f'{true_norm_beam=}, {fit_norm=}, {fit_error=}')
            return num_ps, chi2dof, fit_norm, norm_error, fit_error

        if mode == 'get_num_ps':
            return num_ps, self.num_near_ps, self.ang_near

        if mode == 'check_sigma':
            calc_error()


def see_error_changes():
    from gen_ps import gen_ps
    nside = 1024
    beam = 18
    rlz_idx = 0

    nstd = 1
    flux_idx = 0

    # m = np.load('./data/ps/ps_map_5.npy')
    fit_norm_list = []
    norm_error_list = []
    distance_list = np.linspace(0.1, 1.5, 10)
    for distance_factor in distance_list:
        m = gen_ps(beam=beam, nside=nside, distance_factor=distance_factor, rlz_idx=rlz_idx)

        obj = FitPointSource(m=m, nstd=nstd, nside=nside, flux_idx=flux_idx, radius_factor=1.5, beam=beam, epsilon=0.00001, distance_factor=distance_factor)

        # obj.calc_covariance_matrix(mode='noise')

        _,_,fit_norm,norm_error,_ = obj.fit_all(cov_mode='noise')
        print(f'{distance_factor=}, {fit_norm=}, {norm_error=}')

        fit_norm_list.append(norm_error)


    plt.plot(distance_list, fit_norm_list)
    plt.show()
        # path_params = Path(f'./parameter/two_0.2')
        # path_params.mkdir(exist_ok=True, parents=True)
        # np.save(path_params / Path(f'{rlz_idx}.npy'), fit_norm)

def main():
    from gen_ps import gen_ps
    nside = 1024
    beam = 30
    rlz_idx = 3

    nstd = 1
    flux_idx = 0

    # m = np.load('./data/ps/ps_map_5.npy')
    # distance_list = np.linspace(0.1, 1.5, 10)
    distance_factor = 0.5
    m = gen_ps(beam=beam, nside=nside, distance_factor=distance_factor, rlz_idx=rlz_idx)

    obj = FitPointSource(m=m, nstd=nstd, nside=nside, flux_idx=flux_idx, radius_factor=1.5, beam=beam, epsilon=0.00001, distance_factor=distance_factor)

    obj.calc_covariance_matrix(mode='noise')

    _,_,fit_norm,norm_error,_ = obj.fit_all(cov_mode='noise')
    print(f'{distance_factor=}, {fit_norm=}, {norm_error=}')



    # path_params = Path(f'./parameter/two_0.2')
    # path_params.mkdir(exist_ok=True, parents=True)
    # np.save(path_params / Path(f'{rlz_idx}.npy'), fit_norm)



if __name__ == '__main__':
    # see_error_changes()
    main()








