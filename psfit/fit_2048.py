import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pandas as pd
import time
import pickle
import os

from pathlib import Path
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

    def calc_C_theta_itp_func(self, lgd_itp_func_pos='../../test/interpolate_cov/lgd_itp_funcs350.pkl'):
        def evaluate_interp_func(l, x, interp_funcs):
            for interp_func, x_range in interp_funcs[l]:
                if x_range[0] <= x <= x_range[1]:
                    return interp_func(x)
            raise ValueError(f"x = {x} is out of the interpolation range for l = {l}")

        def calc_C_theta_itp(x, lmax, cl, itp_funcs):
            Pl = np.zeros(lmax+1)
            for l in range(lmax+1):
                Pl[l] = evaluate_interp_func(l, x, interp_funcs=itp_funcs)
            ell = np.arange(lmax+1)
            sum_val = 1 / (4 * np.pi) * np.sum((2 * ell + 1) * cl * Pl)
            print(f'{sum_val=}')
            return sum_val

        with open(lgd_itp_func_pos, 'rb') as f:
            loaded_itp_funcs = pickle.load(f)
        cos_theta_list = np.linspace(0.99, 1, 5000)
        C_theta_list = []
        time0 = time.time()
        for cos_theta in cos_theta_list:
            C_theta = calc_C_theta_itp(x=cos_theta, lmax=self.lmax, cl=self.cl_cmb[0:self.lmax+1], itp_funcs=loaded_itp_funcs)
            C_theta_list.append(C_theta)
        print(f'{C_theta_list=}')
        timecov = time.time()-time0
        print(f'{timecov=}')

        self.cs = CubicSpline(cos_theta_list, C_theta_list)
        if not os.path.exists('./cs'):
            os.makedirs('./cs')
        with open('./cs/cs.pkl', 'wb') as f:
            pickle.dump(self.cs, f)
        return self.cs

    def calc_C_theta(self,):
        if not hasattr(self, "cs"):
            with open('./cs/cs.pkl', 'rb') as f:
                self.cs = pickle.load(f)
            print('cs is ok')

        ipix_fit = self.ipix_fit
        nside = self.nside
        n_cov = self.ndof

        cov = np.zeros((n_cov, n_cov))
        print(f'{cov.shape=}')

        time0 = time.time()

        vec = np.asarray(hp.pix2vec(nside=self.nside, ipix=ipix_fit))
        cos_theta = vec.T@vec
        print(f'{cos_theta.shape=}')
        cov = self.cs(cos_theta)
        print(f'{cov.shape=}')

        timecov = time.time()-time0
        print(f'{timecov=}')
        print(f'{cov=}')
        save_path = f'./cmb_cov_{self.nside}/r_{self.radius_factor}'
        Path(save_path).mkdir(parents=True, exist_ok=True)
        np.save(Path(save_path) / Path(f'{self.flux_idx}.npy'), cov)

    def calc_covariance_matrix(self, mode='cmb+noise'):

        if mode == 'noise':
            nstd2 = (self.nstd**2)[self.ipix_fit]
            cov = np.zeros((self.ndof,self.ndof))

            for i in range(self.ndof):
                cov[i,i] = cov[i,i] + nstd2[i]
            print(f'{cov=}')
            self.inv_cov = np.linalg.inv(cov)
            path_inv_cov = Path(f'inv_cov_{self.nside}/r_{self.radius_factor}') / Path(mode)
            path_inv_cov.mkdir(parents=True, exist_ok=True)
            np.save(path_inv_cov / Path(f'{self.flux_idx}.npy'), self.inv_cov)
            return None

        cmb_cov_path = Path(f'./cmb_cov_{self.nside}/r_{self.radius_factor}') / Path(f'{self.flux_idx}.npy')
        print(f'{cmb_cov_path=}')

        cov = np.load(cmb_cov_path)
        cov  = cov + self.epsilon * np.eye(cov.shape[0])

        if mode == 'cmb':
            # self.inv_cov = np.linalg.inv(cov)
            self.inv_cov = np.linalg.solve(cov, np.eye(cov.shape[0]))
            path_inv_cov = Path(f'inv_cov_{self.nside}/r_{self.radius_factor}') / Path(mode)
            path_inv_cov.mkdir(parents=True, exist_ok=True)
            np.save(path_inv_cov / Path(f'{self.flux_idx}.npy'), self.inv_cov)
            return None

        if mode == 'cmb+noise':
            nstd2 = (self.nstd**2)[self.ipix_fit]
            for i in range(self.ndof):
                cov[i,i] = cov[i,i] + nstd2[i]
            # self.inv_cov = np.linalg.inv(cov)
            self.inv_cov = np.linalg.solve(cov, np.eye(cov.shape[0]))
            path_inv_cov = Path(f'inv_cov_{self.nside}/r_{self.radius_factor}') / Path(mode)
            path_inv_cov.mkdir(parents=True, exist_ok=True)
            np.save(path_inv_cov / Path(f'{self.flux_idx}.npy'), self.inv_cov)
            return None


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

    def find_nearby_ps(self, num_ps=1):
        threshold_factor = self.radius_factor + 1.1
        print(f'{threshold_factor=}')
        dir_0 = (self.lon, self.lat)
        arr_1 = self.df_ps.loc[:, 'Unnamed: 0']
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

        print(f'{index_near[0].shape=}')
        print(f'{ang_near.shape=}')


        # if index_near[0].size == 0:
        #     raise ValueError('This is a single point source, please check! 4 parameter fit should get good fitting result')
        print(f'number of ir, radio ps = {index_near[0].size}')

        lon_list = []
        lat_list = []
        iflux_list = []
        for i in range(min(num_ps, len(index_near[0]))):
            index = index_near[0][i]
            if index < self.df_mask.at[self.flux_idx, 'Unnamed: 0']:
                lon = np.rad2deg(self.df_ps.at[index, 'lon'])
                lat = np.rad2deg(self.df_ps.at[index, 'lat'])
                iflux = self.flux2norm_beam(self.df_ps.at[index, 'iflux'])
            else:
                lon = np.rad2deg(self.df_ps.at[index + 1, 'lon'])
                lat = np.rad2deg(self.df_ps.at[index + 1, 'lat'])
                iflux = self.flux2norm_beam(self.df_ps.at[index + 1, 'iflux'])
            # lon, lat = self.input_lonlat2pix_lonlat(lon, lat)
            lon_list.append(lon)
            lat_list.append(lat)
            iflux_list.append(iflux)

        print(f'{iflux_list=}')
    
        ##Optional visualization code commented out
        #hp.gnomview(self.m, rot=[self.lon,self.lat,0])
        #for lon, lat in zip(lon_list, lat_list):
        #    hp.projscatter(lon, lat, lonlat=True)
        #plt.show()
    
        # return tuple(iflux_list + lon_list + lat_list)
        iflux_arr = np.array(iflux_list)
        num_ps = np.count_nonzero(np.where(iflux_arr > self.flux2norm_beam(flux=100), iflux_arr, 0))
        print(f'there are {num_ps} ps > 100 mJy')

        if num_ps > 0:
            ang_near_and_bigger_than_threshold = ang_near[0:num_ps]
            if any(ang_near_and_bigger_than_threshold < 0.5):
                self.flag_too_near = True

            print(f'{self.flag_too_near = }')

        return num_ps, tuple(sum(zip(iflux_list, lon_list, lat_list), ()))

    def fit_all(self, cov_mode:str, mode:str='pipeline'):
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

            error_estimate = np.sum(y_model**2 / y_err**2)
            print(f"{error_estimate=}")

            z = (y_diff) @ self.inv_cov @ (y_diff)
            return z

        def lsq_params(*args):
            # args is expected to be in the format:
            # norm_beam1, ctr1_lon_shift, ctr1_lat_shift, ..., norm_beamN, ctrN_lon_shift, ctrN_lat_shift, const
        
            num_ps = (len(args) - 1) // 3  # Determine the number of point sources based on the number of arguments
        
            # Extract const
            const = args[-1]
        
            # Process each point source
            thetas = []
            for i in range(num_ps):
                norm_beam, lon_shift, lat_shift = args[i*3:i*3+3]
                # print(f'{lon_shift=},{lat_shift}')
                lon = self.fit_lon[i] + lon_shift
                lat = self.fit_lat[i] + lat_shift
                # print(f'{lon=},{lat=}')
                ctr_vec = np.array(hp.ang2vec(theta=lon, phi=lat, lonlat=True))
        
                theta = hp.rotator.angdist(dir1=ctr_vec, dir2=vec_around)
                thetas.append(norm_beam / (2 * np.pi * self.sigma**2) * np.exp(- (theta)**2 / (2 * self.sigma**2)))
        
            def model():
                return sum(thetas) + const
        
            y_model = model()
            y_data = self.m[ipix_fit]
            y_err = self.nstd[ipix_fit]
        
            y_diff = y_data - y_model

            z = (y_diff) @ self.inv_cov @ (y_diff)
            return z

        self.inv_cov = np.load(f'./inv_cov_{self.nside}/{cov_mode}/{self.flux_idx}.npy')

        ctr0_vec = hp.ang2vec(theta=self.lon, phi=self.lat, lonlat=True)

        ipix_fit = self.ipix_fit
        vec_around = self.vec_around

        num_ps, near = self.find_nearby_ps(num_ps=10)
        print(f'{num_ps=}, {near=}')

        true_norm_beam = self.flux2norm_beam(self.iflux)

        def fit_2_params():
            obj_minuit = Minuit(lsq_2_params, norm_beam=self.ini_norm_beam, const=0.0)
            obj_minuit.limits = [(0,1),  (-100,100)]
            print(obj_minuit.migrad())
            # for p in obj_minuit.params:
                # print(repr(p))

            chi2dof = obj_minuit.fval / self.ndof
            str_chi2 = f"𝜒²/ndof = {obj_minuit.fval:.2f} / {self.ndof} = {chi2dof}"
            print(str_chi2)

            if obj_minuit.fmin.hesse_failed:
                raise ValueError('hesse failed!')
            else:
                return chi2dof, obj_minuit.values['norm_beam'],obj_minuit.errors['norm_beam'], self.lon, self.lat


        def fit_4_params():
            params = (self.ini_norm_beam, 0.0, 0.0, 0.0)
            self.fit_lon = (self.lon,)
            self.fit_lat = (self.lat,)
            print(f'{self.fit_lon=}')

            obj_minuit = Minuit(lsq_params, name=("norm_beam1","ctr1_lon_shift","ctr1_lat_shift","const"), *params)
            obj_minuit.limits = [(0,1.0), (-shift_limit, shift_limit), (-shift_limit,shift_limit), (-100,100)]
            print(obj_minuit.migrad())
            print(obj_minuit.hesse())
            # for p in obj_minuit.params:
                # print(repr(p))

            chi2dof = obj_minuit.fval / self.ndof
            str_chi2 = f"𝜒²/ndof = {obj_minuit.fval:.2f} / {self.ndof} = {chi2dof}"
            print(str_chi2)

            if obj_minuit.fmin.hesse_failed:
                raise ValueError('hesse failed!')

            print(f'4 parameter fitting is enough, hesse ok')
            fit_lon = self.lon + obj_minuit.values['ctr1_lon_shift']
            fit_lat = self.lat + obj_minuit.values['ctr1_lat_shift']
            return chi2dof, obj_minuit.values['norm_beam1'],obj_minuit.errors['norm_beam1'], fit_lon, fit_lat

        def fit_7_params():
            num_ps, (self.ctr2_iflux, self.ctr2_lon, self.ctr2_lat) = self.find_nearby_ps(num_ps=1)

            params = (self.ini_norm_beam, 0.0, 0.0, self.ctr2_iflux, 0, 0,0)
            self.fit_lon = (self.lon, self.ctr2_lon)
            self.fit_lat = (self.lat, self.ctr2_lat)
            obj_minuit = Minuit(lsq_params, name=("norm_beam1","ctr1_lon_shift","ctr1_lat_shift","norm_beam2","ctr2_lon_shift","ctr2_lat_shift","const"), *params)

            obj_minuit.limits = [(0,1), (-shift_limit,shift_limit), (-shift_limit,shift_limit),(0,1),(-shift_limit,shift_limit),(-shift_limit, shift_limit), (-100,100)]
            print(obj_minuit.migrad())
            print(obj_minuit.hesse())
            # for p in obj_minuit.params:
                # print(repr(p))

            chi2dof = obj_minuit.fval / self.ndof
            str_chi2 = f"𝜒²/ndof = {obj_minuit.fval:.2f} / {self.ndof} = {chi2dof}"
            print(str_chi2)

            if obj_minuit.fmin.hesse_failed:
                raise ValueError('hesse failed!')

            print(f'7 parameter fitting is enough, hesse ok')
            fit_lon = self.lon + obj_minuit.values['ctr1_lon_shift']
            fit_lat = self.lat + obj_minuit.values['ctr1_lat_shift']

            return chi2dof, obj_minuit.values['norm_beam1'],obj_minuit.errors['norm_beam1'], fit_lon, fit_lat

        def fit_10_params():
            num_ps, (self.ctr2_iflux, self.ctr2_lon, self.ctr2_lat, self.ctr3_iflux, self.ctr3_lon, self.ctr3_lat) = self.find_nearby_ps(num_ps=2)

            params = (self.ini_norm_beam, 0, 0, self.ctr2_iflux, 0, 0, self.ctr3_iflux, 0, 0, 0)
            self.fit_lon = (self.lon, self.ctr2_lon, self.ctr3_lon)
            self.fit_lat = (self.lat, self.ctr2_lat, self.ctr3_lat)
            obj_minuit = Minuit(lsq_params, name=("norm_beam1","ctr1_lon_shift","ctr1_lat_shift","norm_beam2","ctr2_lon_shift","ctr2_lat_shift","norm_beam3","ctr3_lon_shift","ctr3_lat_shift","const"), *params)

            obj_minuit.limits = [(0,1),(-shift_limit,shift_limit),(-shift_limit,shift_limit),(0,1),(-shift_limit,shift_limit),(-shift_limit,shift_limit),(0,1),(-shift_limit,shift_limit),(-shift_limit,shift_limit), (-100,100)]
            # obj_minuit.errors = (1e-3, 0.01, 0.01, 1e-3, 0.01, 0.01, 1e-3, 0.01, 0.01, 0.1)
            print(obj_minuit.migrad())
            print(obj_minuit.hesse())
            # for p in obj_minuit.params:
                # print(repr(p))

            chi2dof = obj_minuit.fval / self.ndof
            str_chi2 = f"𝜒²/ndof = {obj_minuit.fval:.2f} / {self.ndof} = {chi2dof}"
            print(str_chi2)

            if obj_minuit.fmin.hesse_failed:
                raise ValueError('hesse failed!')

            print(f'10 parameter fitting is enough, hesse ok')
            fit_lon = self.lon + obj_minuit.values['ctr1_lon_shift']
            fit_lat = self.lat + obj_minuit.values['ctr1_lat_shift']
            return chi2dof, obj_minuit.values['norm_beam1'],obj_minuit.errors['norm_beam1'], fit_lon, fit_lat

        def fit_13_params():
            num_ps, (self.ctr2_iflux, self.ctr2_lon, self.ctr2_lat, self.ctr3_iflux, self.ctr3_lon, self.ctr3_lat, self.ctr4_iflux, self.ctr4_lon, self.ctr4_lat) = self.find_nearby_ps(num_ps=3)
            params = (self.ini_norm_beam, 0, 0, self.ctr2_iflux, 0, 0, self.ctr3_iflux, 0, 0, self.ctr4_iflux, 0, 0, 0)

            self.fit_lon = (self.lon, self.ctr2_lon, self.ctr3_lon, self.ctr4_lon)
            self.fit_lat = (self.lat, self.ctr2_lat, self.ctr3_lat, self.ctr4_lat)
            obj_minuit = Minuit(lsq_params, name=("norm_beam1","ctr1_lon_shift","ctr1_lat_shift","norm_beam2","ctr2_lon_shift","ctr2_lat_shift","norm_beam3","ctr3_lon_shift","ctr3_lat_shift","norm_beam4","ctr4_lon_shift","ctr4_lat_shift","const"), *params)

            # obj_minuit.limits = [(0,1),(-0.3,0.3),(-0.3,0.3),(0,1),(-0.4,0.4),(-0.4,0.4),(0,1),(-0.5,0.5),(-0.5,0.5), (0,1),(-0.5,0.5),(-0.5,0.5), (-100,100)]

            obj_minuit.limits = [(0,1),(-shift_limit,shift_limit),(-shift_limit,shift_limit),(0,1),(-shift_limit,shift_limit),(-shift_limit,shift_limit),(0,1),(-shift_limit,shift_limit),(-shift_limit,shift_limit), (0,1),(-shift_limit, shift_limit),(-shift_limit, shift_limit), (-100,100)]
            print(obj_minuit.migrad())
            print(obj_minuit.hesse())
            # for p in obj_minuit.params:
                # print(repr(p))

            chi2dof = obj_minuit.fval / self.ndof
            str_chi2 = f"𝜒²/ndof = {obj_minuit.fval:.2f} / {self.ndof} = {chi2dof}"
            print(str_chi2)

            is_ps = obj_minuit.values['norm_beam1'] > self.sigma_threshold * obj_minuit.errors['norm_beam1']

            if obj_minuit.fmin.hesse_failed:
                raise ValueError('hesse failed!')

            print(f'13 parameter fitting is enough, hesse ok')
            fit_lon = self.lon + obj_minuit.values['ctr1_lon_shift']
            fit_lat = self.lat + obj_minuit.values['ctr1_lat_shift']

            return chi2dof, obj_minuit.values['norm_beam1'],obj_minuit.errors['norm_beam1'], fit_lon, fit_lat

        def fit_16_params():
            num_ps, (self.ctr2_iflux, self.ctr2_lon, self.ctr2_lat, self.ctr3_iflux, self.ctr3_lon, self.ctr3_lat, self.ctr4_iflux, self.ctr4_lon, self.ctr4_lat, self.ctr5_iflux, self.ctr5_lon, self.ctr5_lat) = self.find_nearby_ps(num_ps=4)

            params = (self.ini_norm_beam, 0, 0, self.ctr2_iflux, 0, 0, self.ctr3_iflux, 0, 0, self.ctr4_iflux, 0, 0, self.ctr5_iflux,0,0,0)
            self.fit_lon = (self.lon, self.ctr2_lon, self.ctr3_lon, self.ctr4_lon, self.ctr5_lon)
            self.fit_lat = (self.lat, self.ctr2_lat, self.ctr3_lat, self.ctr4_lat, self.ctr5_lat)
            obj_minuit = Minuit(lsq_params, name=("norm_beam1","ctr1_lon_shift","ctr1_lat_shift","norm_beam2","ctr2_lon_shift","ctr2_lat_shift","norm_beam3","ctr3_lon_shift","ctr3_lat_shift","norm_beam4","ctr4_lon_shift","ctr4_lat_shift","norm_beam5","ctr5_lon_shift","ctr5_lat_shift","const"), *params)

            # obj_minuit.limits = [(0,1),(-0.3,0.3),(-0.3,0.3),(0,1),(-0.3,0.3),(-0.3,0.3),(0,1),(-0.5,0.5),(-0.5,0.5), (0,1),(-0.3,0.3),(-0.3,0.3),(0,1),(-0.5,0.5),(-0.5,0.5),(-100,100)]

            obj_minuit.limits = [(0,1),(-shift_limit,shift_limit),(-shift_limit,shift_limit),(0,1),(-shift_limit,shift_limit),(-shift_limit,shift_limit),(0,1),(-shift_limit,shift_limit),(-shift_limit,shift_limit), (0,1),(-shift_limit, shift_limit),(-shift_limit, shift_limit),(0,1),(-shift_limit, shift_limit),(-shift_limit, shift_limit), (-100,100)]
            print(obj_minuit.migrad())
            print(obj_minuit.hesse())
            # for p in obj_minuit.params:
                # print(repr(p))

            chi2dof = obj_minuit.fval / self.ndof
            str_chi2 = f"𝜒²/ndof = {obj_minuit.fval:.2f} / {self.ndof} = {chi2dof}"
            print(str_chi2)

            if obj_minuit.fmin.hesse_failed:
                raise ValueError('hesse failed!')

            print(f'16 parameter fitting is enough, hesse ok')
            fit_lon = self.lon + obj_minuit.values['ctr1_lon_shift']
            fit_lat = self.lon + obj_minuit.values['ctr1_lat_shift']

            return chi2dof, obj_minuit.values['norm_beam1'], obj_minuit.errors['norm_beam1'], fit_lon, fit_lat

        def fit_19_params():
            num_ps, (self.ctr2_iflux, self.ctr2_lon, self.ctr2_lat, self.ctr3_iflux, self.ctr3_lon, self.ctr3_lat, self.ctr4_iflux, self.ctr4_lon, self.ctr4_lat, self.ctr5_iflux, self.ctr5_lon, self.ctr5_lat, self.ctr6_iflux, self.ctr6_lon, self.ctr6_lat) = self.find_nearby_ps(num_ps=5)

            params = (self.ini_norm_beam, 0, 0, self.ctr2_iflux, 0, 0, self.ctr3_iflux, 0, 0, self.ctr4_iflux, 0, 0, self.ctr5_iflux,0,0, self.ctr6_iflux, 0, 0, 0)
            self.fit_lon = (self.lon, self.ctr2_lon, self.ctr3_lon, self.ctr4_lon, self.ctr5_lon, self.ctr6_lon)
            self.fit_lat = (self.lat, self.ctr2_lat, self.ctr3_lat, self.ctr4_lat, self.ctr5_lat, self.ctr6_lat)
            obj_minuit = Minuit(lsq_params, name=("norm_beam1","ctr1_lon_shift","ctr1_lat_shift","norm_beam2","ctr2_lon_shift","ctr2_lat_shift","norm_beam3","ctr3_lon_shift","ctr3_lat_shift","norm_beam4","ctr4_lon_shift","ctr4_lat_shift","norm_beam5","ctr5_lon_shift","ctr5_lat_shift","norm_beam6","ctr6_lon_shift","ctr6_lat_shift","const"), *params)

            # obj_minuit.limits = [(0,1),(-0.3,0.3),(-0.3,0.3),(0,1),(-0.3,0.3),(-0.3,0.3),(0,1),(-0.5,0.5),(-0.5,0.5), (0,1),(-0.3,0.3),(-0.3,0.3),(0,1),(-0.5,0.5),(-0.5,0.5),(-100,100)]

            obj_minuit.limits = [(0,1),(-shift_limit,shift_limit),(-shift_limit,shift_limit),(0,1),(-shift_limit,shift_limit),(-shift_limit,shift_limit),(0,1),(-shift_limit,shift_limit),(-shift_limit,shift_limit), (0,1),(-shift_limit, shift_limit),(-shift_limit, shift_limit),(0,1),(-shift_limit, shift_limit),(-shift_limit, shift_limit),(0,1),(-shift_limit, shift_limit),(-shift_limit, shift_limit), (-100,100)]
            print(obj_minuit.migrad())
            print(obj_minuit.hesse())
            # for p in obj_minuit.params:
                # print(repr(p))

            chi2dof = obj_minuit.fval / self.ndof
            str_chi2 = f"𝜒²/ndof = {obj_minuit.fval:.2f} / {self.ndof} = {chi2dof}"
            print(str_chi2)

            if obj_minuit.fmin.hesse_failed:
                raise ValueError('hesse failed!')

            print(f'19 parameter fitting is enough, hesse ok')
            fit_lon = self.lon + obj_minuit.values['ctr1_lon_shift']
            fit_lat = self.lon + obj_minuit.values['ctr1_lat_shift']

            return chi2dof, obj_minuit.values['norm_beam1'], obj_minuit.errors['norm_beam1'], fit_lon, fit_lat

        def fit_22_params():
            num_ps, (self.ctr2_iflux, self.ctr2_lon, self.ctr2_lat, self.ctr3_iflux, self.ctr3_lon, self.ctr3_lat, self.ctr4_iflux, self.ctr4_lon, self.ctr4_lat, self.ctr5_iflux, self.ctr5_lon, self.ctr5_lat, self.ctr6_iflux, self.ctr6_lon, self.ctr6_lat, self.ctr7_iflux, self.ctr7_lon, self.ctr7_lat) = self.find_nearby_ps(num_ps=6)

            params = (self.ini_norm_beam, 0, 0, self.ctr2_iflux, 0, 0, self.ctr3_iflux, 0, 0, self.ctr4_iflux, 0, 0, self.ctr5_iflux,0,0, self.ctr6_iflux, 0, 0, self.ctr7_iflux, 0, 0, 0)
            self.fit_lon = (self.lon, self.ctr2_lon, self.ctr3_lon, self.ctr4_lon, self.ctr5_lon, self.ctr6_lon, self.ctr7_lon)
            self.fit_lat = (self.lat, self.ctr2_lat, self.ctr3_lat, self.ctr4_lat, self.ctr5_lat, self.ctr6_lat, self.ctr7_lat)
            obj_minuit = Minuit(lsq_params, name=("norm_beam1","ctr1_lon_shift","ctr1_lat_shift","norm_beam2","ctr2_lon_shift","ctr2_lat_shift","norm_beam3","ctr3_lon_shift","ctr3_lat_shift","norm_beam4","ctr4_lon_shift","ctr4_lat_shift","norm_beam5","ctr5_lon_shift","ctr5_lat_shift","norm_beam6","ctr6_lon_shift","ctr6_lat_shift","norm_beam7","ctr7_lon_shift","ctr7_lat_shift","const"), *params)

            # obj_minuit.limits = [(0,1),(-0.3,0.3),(-0.3,0.3),(0,1),(-0.3,0.3),(-0.3,0.3),(0,1),(-0.5,0.5),(-0.5,0.5), (0,1),(-0.3,0.3),(-0.3,0.3),(0,1),(-0.5,0.5),(-0.5,0.5),(-100,100)]

            obj_minuit.limits = [(0,1),(-shift_limit,shift_limit),(-shift_limit,shift_limit),(0,1),(-shift_limit,shift_limit),(-shift_limit,shift_limit),(0,1),(-shift_limit,shift_limit),(-shift_limit,shift_limit), (0,1),(-shift_limit, shift_limit),(-shift_limit, shift_limit),(0,1),(-shift_limit, shift_limit),(-shift_limit, shift_limit),(0,1),(-shift_limit, shift_limit),(-shift_limit, shift_limit),(0,1),(-shift_limit, shift_limit),(-shift_limit, shift_limit), (-100,100)]
            print(obj_minuit.migrad())
            print(obj_minuit.hesse())
            # for p in obj_minuit.params:
                # print(repr(p))

            chi2dof = obj_minuit.fval / self.ndof
            str_chi2 = f"𝜒²/ndof = {obj_minuit.fval:.2f} / {self.ndof} = {chi2dof}"
            print(str_chi2)

            if obj_minuit.fmin.hesse_failed:
                raise ValueError('hesse failed!')

            print(f'22 parameter fitting is enough, hesse ok')
            fit_lon = self.lon + obj_minuit.values['ctr1_lon_shift']
            fit_lat = self.lon + obj_minuit.values['ctr1_lat_shift']

            return chi2dof, obj_minuit.values['norm_beam1'], obj_minuit.errors['norm_beam1'], fit_lon, fit_lat

        def fit_25_params():
            num_ps, (self.ctr2_iflux, self.ctr2_lon, self.ctr2_lat, self.ctr3_iflux, self.ctr3_lon, self.ctr3_lat, self.ctr4_iflux, self.ctr4_lon, self.ctr4_lat, self.ctr5_iflux, self.ctr5_lon, self.ctr5_lat, self.ctr6_iflux, self.ctr6_lon, self.ctr6_lat, self.ctr7_iflux, self.ctr7_lon, self.ctr7_lat, self.ctr8_iflux, self.ctr8_lon, self.ctr8_lat) = self.find_nearby_ps(num_ps=7)

            params = (self.ini_norm_beam, 0, 0, self.ctr2_iflux, 0, 0, self.ctr3_iflux, 0, 0, self.ctr4_iflux, 0, 0, self.ctr5_iflux,0,0, self.ctr6_iflux, 0, 0, self.ctr7_iflux, 0, 0,self.ctr8_iflux, 0, 0, 0)
            self.fit_lon = (self.lon, self.ctr2_lon, self.ctr3_lon, self.ctr4_lon, self.ctr5_lon, self.ctr6_lon, self.ctr7_lon, self.ctr8_lon)
            self.fit_lat = (self.lat, self.ctr2_lat, self.ctr3_lat, self.ctr4_lat, self.ctr5_lat, self.ctr6_lat, self.ctr7_lat, self.ctr8_lon)
            obj_minuit = Minuit(lsq_params, name=("norm_beam1","ctr1_lon_shift","ctr1_lat_shift","norm_beam2","ctr2_lon_shift","ctr2_lat_shift","norm_beam3","ctr3_lon_shift","ctr3_lat_shift","norm_beam4","ctr4_lon_shift","ctr4_lat_shift","norm_beam5","ctr5_lon_shift","ctr5_lat_shift","norm_beam6","ctr6_lon_shift","ctr6_lat_shift","norm_beam7","ctr7_lon_shift","ctr7_lat_shift","norm_beam8","ctr8_lon_shift","ctr8_lat_shift","const"), *params)

            # obj_minuit.limits = [(0,1),(-0.3,0.3),(-0.3,0.3),(0,1),(-0.3,0.3),(-0.3,0.3),(0,1),(-0.5,0.5),(-0.5,0.5), (0,1),(-0.3,0.3),(-0.3,0.3),(0,1),(-0.5,0.5),(-0.5,0.5),(-100,100)]

            obj_minuit.limits = [(0,1),(-shift_limit,shift_limit),(-shift_limit,shift_limit),(0,1),(-shift_limit,shift_limit),(-shift_limit,shift_limit),(0,1),(-shift_limit,shift_limit),(-shift_limit,shift_limit), (0,1),(-shift_limit, shift_limit),(-shift_limit, shift_limit),(0,1),(-shift_limit, shift_limit),(-shift_limit, shift_limit),(0,1),(-shift_limit, shift_limit),(-shift_limit, shift_limit),(0,1),(-shift_limit, shift_limit),(-shift_limit, shift_limit),(0,1),(-shift_limit, shift_limit),(-shift_limit, shift_limit), (-100,100)]
            print(obj_minuit.migrad())
            print(obj_minuit.hesse())
            # for p in obj_minuit.params:
                # print(repr(p))

            chi2dof = obj_minuit.fval / self.ndof
            str_chi2 = f"𝜒²/ndof = {obj_minuit.fval:.2f} / {self.ndof} = {chi2dof}"
            print(str_chi2)

            if obj_minuit.fmin.hesse_failed:
                raise ValueError('hesse failed!')

            print(f'25 parameter fitting is enough, hesse ok')
            fit_lon = self.lon + obj_minuit.values['ctr1_lon_shift']
            fit_lat = self.lon + obj_minuit.values['ctr1_lat_shift']

            return chi2dof, obj_minuit.values['norm_beam1'], obj_minuit.errors['norm_beam1'], fit_lon, fit_lat



        print(f'begin point source fitting, first do 2 parameter fit...')
        chi2dof, fit_norm, norm_error, fit_lon, fit_lat = fit_2_params()
        if fit_norm < self.sigma_threshold * norm_error:
            print('there is no point sources.')

        if mode == 'streamline':
            chi2dof, fit_norm, norm_error, fit_lon, fit_lat = fit_4_params()
            if chi2dof < 1.06:
                return fit_norm, norm_error, fit_lon, fit_lat
            else:
                chi2dof, fit_norm, norm_error, fit_lon, fit_lat = fit_7_params()
                if chi2dof < 1.06:
                    return fit_norm, norm_error, fit_lon, fit_lat
                else:
                    chi2dof, fit_norm, norm_error, fit_lon, fit_lat = fit_10_params()
                    if chi2dof < 1.06:
                        return fit_norm, norm_error, fit_lon, fit_lat
                    else:
                        chi2dof, fit_norm, norm_error, fit_lon, fit_lat = fit_13_params()
                        if chi2dof < 1.06:
                            return fit_norm, norm_error, fit_lon, fit_lat
                        else:
                            print('Cannot fit good!!!')
                            return fit_norm, norm_error, fit_lon, fit_lat

        if mode == 'pipeline':
            shift_limit = 0.03
            if self.flag_too_near == True:
                shift_limit = 0.02

            if num_ps == 0:
                chi2dof, fit_norm, norm_error, fit_lon, fit_lat = fit_4_params()
            elif num_ps == 1:
                chi2dof, fit_norm, norm_error, fit_lon, fit_lat = fit_7_params()
            elif num_ps == 2:
                chi2dof, fit_norm, norm_error, fit_lon, fit_lat = fit_10_params()
            elif num_ps == 3:
                chi2dof, fit_norm, norm_error, fit_lon, fit_lat = fit_13_params()
            elif num_ps == 4:
                chi2dof, fit_norm, norm_error, fit_lon, fit_lat = fit_16_params()
            elif num_ps == 5:
                chi2dof, fit_norm, norm_error, fit_lon, fit_lat = fit_19_params()
            elif num_ps == 6:
                chi2dof, fit_norm, norm_error, fit_lon, fit_lat = fit_22_params()
            elif num_ps == 7:
                chi2dof, fit_norm, norm_error, fit_lon, fit_lat = fit_25_params()


            fit_error = np.abs(fit_norm - true_norm_beam) / true_norm_beam

            print(f'{num_ps=}, {chi2dof=}, {fit_norm=}, {norm_error=}, {fit_lon=}, {fit_lat=}')
            print(f'{true_norm_beam=}, {fit_norm=}, {fit_error=}')
            return num_ps, chi2dof, fit_norm, norm_error, fit_lon, fit_lat, fit_error




if __name__ == '__main__':
    # m = np.load('../../fitdata/synthesis_data/2048/PSNOISE/40/1.npy')[0]
    m = np.load('../../fitdata/synthesis_data/2048/PSCMBFGNOISE/40/2.npy')[0]
    nstd = np.load('../../FGSim/NSTDNORTH/2048/40.npy')[0]
    df_mask = pd.read_csv('../partial_sky_ps/ps_in_mask/2048/40mask.csv')
    flux_idx = 4
    lon = np.rad2deg(df_mask.at[flux_idx, 'lon'])
    lat = np.rad2deg(df_mask.at[flux_idx, 'lat'])
    iflux = df_mask.at[flux_idx, 'iflux']

    df_ps = pd.read_csv('../partial_sky_ps/ps_in_mask/2048/40ps.csv')
    
    lmax = 350
    nside = 2048
    beam = 63
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

    obj = FitPointSource(m=m, nstd=nstd, flux_idx=flux_idx, df_mask=df_mask, df_ps=df_ps, cl_cmb=cl_cmb, lon=lon, lat=lat, iflux=iflux, lmax=lmax, nside=nside, radius_factor=1.5, beam=beam, epsilon=1e-5)

    # obj.see_true_map(m=m, lon=lon, lat=lat, nside=nside, beam=beam)

    # obj.calc_covariance_matrix(mode='noise', cmb_cov_fold='../cmb_cov_calc/cov')

    # obj.calc_C_theta_itp_func(lgd_itp_func_pos='../../test/interpolate_cov/lgd_itp_funcs350.pkl')
    # obj.calc_C_theta(save_path='./cov_r_2.0/2048')
    # obj.calc_precise_C_theta()

    obj.calc_C_theta()
    obj.calc_covariance_matrix(mode='cmb+noise')

    # obj.fit_all(cov_mode='cmb+noise')


