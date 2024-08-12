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

class FitPolPS:
    @staticmethod
    def mJy_to_uKCMB(intensity_mJy, frequency_GHz):
        # Constants
        c = 2.99792458e8  # Speed of light in m/s
        h = 6.62607015e-34  # Planck constant in J*s
        k = 1.380649e-23  # Boltzmann constant in J/K
        T_CMB = 2.725  # CMB temperature in Kelvin

        frequency_Hz = frequency_GHz * 1e9 # Convert frequency to Hz from GHz

        x = (h * frequency_Hz) / (k * T_CMB) # Calculate x = h*nu/(k*T)

        # Calculate the derivative of the Planck function with respect to temperature, dB/dT
        dBdT = (2.0 * h * frequency_Hz**3 / c**2 / T_CMB) * (x * np.exp(x) / (np.exp(x) - 1)**2)
        intensity_Jy = intensity_mJy * 1e-3 # Convert intensity from mJy to Jy
        intensity_W_m2_sr_Hz = intensity_Jy * 1e-26 # Convert Jy/sr to W/m^2/sr/Hz
        uK_CMB = intensity_W_m2_sr_Hz / dBdT * 1e6 # Convert to uK_CMB, taking the inverse of dB/dT
        return uK_CMB

    @staticmethod
    def calculate_P_error(Q, U, sigma_Q, sigma_U):
        P = np.sqrt(Q**2 + U**2)

        partial_P_Q = Q / np.sqrt(Q**2 + U**2)
        partial_P_U = U / np.sqrt(Q**2 + U**2)

        sigma_P = np.sqrt((partial_P_Q * sigma_Q)**2 + (partial_P_U * sigma_U)**2)
        return P, sigma_P

    @staticmethod
    def calculate_phi_error(Q, U, sigma_Q, sigma_U):
        phi = np.arctan2(U, Q)

        partial_phi_Q = -U / (Q**2 + U**2)
        partial_phi_U = Q / (Q**2 + U**2)

        sigma_phi = np.sqrt((partial_phi_Q * sigma_Q)**2 + (partial_phi_U * sigma_U)**2)

        return phi, sigma_phi

    def __init__(self, m_q, m_u, freq, nstd_q, nstd_u, flux_idx, df_mask, df_ps, lmax, nside, radius_factor, beam, sigma_threshold=5, epsilon=1e-4, debug_flag=False):
        self.m_q = m_q # sky maps (npix,)
        self.m_u = m_u # sky maps (npix,)
        self.freq = freq # frequency
        self.df_mask = df_mask # pandas data frame of point sources in mask
        self.lon_rad = df_mask.at[flux_idx, 'lon'] # longitude of the point sources in rad
        self.lat_rad = df_mask.at[flux_idx, 'lat'] # latitude of the point sources in rad
        self.lon = np.rad2deg(self.lon_rad) # latitude of the point sources in degree
        self.lat = np.rad2deg(self.lat_rad) # latitude of the point sources in degree
        self.iflux = df_mask.at[flux_idx, 'iflux']
        self.qflux = df_mask.at[flux_idx, 'qflux']
        self.uflux = df_mask.at[flux_idx, 'uflux']
        self.flux_idx = flux_idx # index in df_mask
        self.df_ps = df_ps # pandas data frame of all point sources
        self.nstd_q = nstd_q # noise standard deviation of Q
        self.nstd_u = nstd_u # noise standard deviation of U
        self.lmax = lmax # maximum multipole
        self.nside = nside # resolution of healpy maps
        self.radius_factor = radius_factor # disc radius of fitting region
        self.sigma_threshold = sigma_threshold # judge if a signal is a point source
        self.beam = beam # in arcmin
        self.epsilon = epsilon # if CMB covariance matrix is not semi-positive, add this to cross term
        self.nside2pixarea_factor = hp.nside2pixarea(nside=self.nside)

        self.i_amp = self.flux2norm_beam(self.iflux) / self.nside2pixarea_factor
        self.q_amp = self.flux2norm_beam(self.qflux) / self.nside2pixarea_factor
        self.u_amp = self.flux2norm_beam(self.uflux) / self.nside2pixarea_factor
        self.p_amp = np.sqrt(self.q_amp**2 + self.u_amp**2)
        self.phi = np.arctan2(self.u_amp, self.q_amp)

        self.sigma = np.deg2rad(beam) / 60 / (np.sqrt(8 * np.log(2)))

        ctr0_pix = hp.ang2pix(nside=self.nside, theta=self.lon, phi=self.lat, lonlat=True)
        self.ctr0_vec = np.array(hp.pix2vec(nside=self.nside, ipix=ctr0_pix)).astype(np.float64)

        self.ipix_fit = hp.query_disc(nside=self.nside, vec=self.ctr0_vec, radius=self.radius_factor * np.deg2rad(self.beam) / 60)
        path_pix_idx = Path(f'./pix_idx_qu')
        path_pix_idx.mkdir(exist_ok=True, parents=True)
        np.save(path_pix_idx / Path(f'{self.flux_idx}.npy'), self.ipix_fit)

        ## if you want the ipix_fit to range from near point to far point, add the following code
        # self.vec_around = np.array(hp.pix2vec(nside=self.nside, ipix=self.ipix_fit.astype(int))).astype(np.float64)
        # angle = np.rad2deg(hp.rotator.angdist(dir1=self.ctr0_vec, dir2=self.vec_around))
        # logger.debug(f'{angle=}')
        # sorted_idx = np.argsort(angle)
        # logger.debug(f'{self.ipix_fit=}')
        # self.ipix_fit = self.ipix_fit[sorted_idx]
        # logger.debug(f'{self.ipix_fit=}')

        self.vec_around = np.array(hp.pix2vec(nside=self.nside, ipix=self.ipix_fit.astype(int))).astype(np.float64)
        _, self.phi_around = hp.pix2ang(nside=self.nside, ipix=self.ipix_fit)
        self.ndof = len(self.ipix_fit) * 2

        self.num_near_ps = 0
        self.flag_too_near = False
        self.flag_overlap = False

        logger.info(f'{lmax=}, {nside=}')
        logger.info(f'{freq=}, {beam=}, {flux_idx=}, {radius_factor=}, lon={self.lon}, lat={self.lat}, ndof={self.ndof}')
        logger.info(f'iflux={self.iflux=}, {self.qflux=}, {self.uflux=}')
        logger.info(f'i_amp={self.i_amp}, q_amp={self.q_amp}, u_amp={self.u_amp}, p_amp={self.p_amp}')
    def get_pix_ind(self):
        return self.ipix_fit

    def calc_definite_fixed_cmb_cov(self):

        cmb_cov_path = Path(f'./cmb_qu_cov/{self.flux_idx}.npy')
        # cmb_cov_path = Path(f'./exp_cov_QU.npy')
        cov = np.load(cmb_cov_path)
        logger.debug(f'{cov=}')
        eigenval, eigenvec = np.linalg.eigh(cov)
        logger.debug(f'{eigenval=}')
        eigenval[eigenval < 0] = 1e-6

        reconstructed_cov = np.dot(eigenvec * eigenval, eigenvec.T)
        reconstructed_eigenval,_ = np.linalg.eigh(reconstructed_cov)
        logger.debug(f'{reconstructed_eigenval=}')
        logger.debug(f'{np.max(np.abs(reconstructed_cov-cov))=}')
        semi_def_cmb_cov = Path(f'semi_def_cmb_cov_{self.nside}/r_{self.radius_factor}')
        semi_def_cmb_cov.mkdir(parents=True, exist_ok=True)
        np.save(semi_def_cmb_cov / Path(f'{self.flux_idx}.npy'), reconstructed_cov)

    def calc_covariance_matrix(self, mode='cmb+noise'):

        if mode == 'noise':
            nstd_q2 = (self.nstd_q**2)[self.ipix_fit].copy()
            nstd_u2 = (self.nstd_u**2)[self.ipix_fit].copy()
            nstd2 = np.concatenate([nstd_q2, nstd_u2])
            logger.debug(f'{nstd2.shape=}')

            cov = np.zeros((self.ndof,self.ndof))

            for i in range(self.ndof):
                cov[i,i] = cov[i,i] + nstd2[i]
            logger.debug(f'{cov=}')
            self.inv_cov = np.linalg.inv(cov)
            path_inv_cov = Path(f'inv_cov_{self.nside}/r_{self.radius_factor}') / Path(mode)
            path_inv_cov.mkdir(parents=True, exist_ok=True)
            np.save(path_inv_cov / Path(f'{self.flux_idx}.npy'), self.inv_cov)
            return None

        cmb_cov_path = Path(f'./semi_def_cmb_cov_{self.nside}/r_{self.radius_factor}') / Path(f'{self.flux_idx}.npy')
        logger.info(f'{cmb_cov_path=}')

        cov = np.load(cmb_cov_path)
        logger.debug(f'{cov.shape=}')
        cov  = cov + self.epsilon * np.eye(cov.shape[0])

        if mode == 'cmb':
            # self.inv_cov = np.linalg.inv(cov)
            self.inv_cov = np.linalg.solve(cov, np.eye(cov.shape[0]))
            path_inv_cov = Path(f'inv_cov_{self.nside}/r_{self.radius_factor}') / Path(mode)
            path_inv_cov.mkdir(parents=True, exist_ok=True)
            np.save(path_inv_cov / Path(f'{self.flux_idx}.npy'), self.inv_cov)
            return None

        if mode == 'cmb+noise':
            nstd_q2 = (self.nstd_q**2)[self.ipix_fit].copy()
            nstd_u2 = (self.nstd_u**2)[self.ipix_fit].copy()
            nstd2 = np.concatenate([nstd_q2, nstd_u2])
            logger.debug(f'{nstd2.shape=}')
            logger.debug(f'{cov=}')
            for i in range(self.ndof):
                cov[i,i] = cov[i,i] + nstd2[i]
            logger.debug(f'{nstd2=}')
            logger.debug(f'{cov=}')
            # self.inv_cov = np.linalg.inv(cov)
            self.inv_cov = np.linalg.solve(cov, np.eye(cov.shape[0]))

            I_exp = cov @ self.inv_cov
            print(f'{I_exp=}')
            # self.inv_cov = np.linalg.pinv(cov)
            path_inv_cov = Path(f'inv_cov_{self.nside}/r_{self.radius_factor}') / Path(mode)
            path_inv_cov.mkdir(parents=True, exist_ok=True)
            np.save(path_inv_cov / Path(f'{self.flux_idx}.npy'), self.inv_cov)
            return None


    def flux2norm_beam(self, flux):
        # from mJy to muK_CMB to norm_beam
        coeffmJy2norm = FitPolPS.mJy_to_uKCMB(1, self.freq)
        logger.debug(f'{coeffmJy2norm=}')
        return coeffmJy2norm * flux

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

    def see_true_map(self, m_q, m_u, nside, beam, **kwargs):
        lon = self.lon
        lat = self.lat
        radiops = hp.read_map(f'/sharefs/alicpt/users/zrzhang/allFreqPSMOutput/skyinbands/AliCPT_uKCMB/{self.freq}GHz/strongradiops_map_{self.freq}GHz.fits', field=0)
        irps = hp.read_map(f'/sharefs/alicpt/users/zrzhang/allFreqPSMOutput/skyinbands/AliCPT_uKCMB/{self.freq}GHz/strongirps_map_{self.freq}GHz.fits', field=0)

        hp.gnomview(irps, rot=[lon, lat, 0], xsize=100, ysize=100, reso=1, title='irps', sub=223)
        hp.gnomview(radiops, rot=[lon, lat, 0], xsize=100, ysize=100, reso=1, title='radiops', sub=224)
        hp.gnomview(m_q, rot=[lon, lat, 0], xsize=100, ysize=100, sub=221, title='Q map')
        hp.gnomview(m_u, rot=[lon, lat, 0], xsize=100, ysize=100, sub=222, title='U map')
        plt.show()

        vec = hp.ang2vec(theta=lon, phi=lat, lonlat=True)
        ipix_disc = hp.query_disc(nside=nside, vec=vec, radius=np.deg2rad(beam)/60)

        mask = np.ones(hp.nside2npix(nside))
        mask[ipix_disc] = 0

        hp.gnomview(mask, rot=[lon, lat, 0])
        plt.show()

    def find_nearby_ps(self, num_ps=1, threshold_extra_factor=1.1):
        threshold_factor = self.radius_factor + threshold_extra_factor
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
        logger.debug(f'{threshold=}')
        logger.debug(f'{ang=}')

        if len(ang) - np.count_nonzero(ang) > 0:
            logger.debug(f'there are some point sources overlap')
            self.flag_overlap = True
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
        pflux_list = []
        qflux_list = []
        uflux_list = []
        for i in range(min(num_ps, len(index_near[0]))):
            index = index_near[0][i]
            if index < self.df_mask.at[self.flux_idx, 'flux_idx']:
                lon = np.rad2deg(self.df_ps.at[index, 'lon'])
                lat = np.rad2deg(self.df_ps.at[index, 'lat'])
                pflux = self.flux2norm_beam(self.df_ps.at[index, 'pflux']) / self.nside2pixarea_factor
                qflux = self.flux2norm_beam(self.df_ps.at[index, 'qflux']) / self.nside2pixarea_factor
                uflux = self.flux2norm_beam(self.df_ps.at[index, 'uflux']) / self.nside2pixarea_factor
            else:
                lon = np.rad2deg(self.df_ps.at[index + 1, 'lon'])
                lat = np.rad2deg(self.df_ps.at[index + 1, 'lat'])
                pflux = self.flux2norm_beam(self.df_ps.at[index + 1, 'pflux']) / self.nside2pixarea_factor
                qflux = self.flux2norm_beam(self.df_ps.at[index + 1, 'qflux']) / self.nside2pixarea_factor
                uflux = self.flux2norm_beam(self.df_ps.at[index + 1, 'uflux']) / self.nside2pixarea_factor
            lon_list.append(lon)
            lat_list.append(lat)
            pflux_list.append(pflux)
            qflux_list.append(qflux)
            uflux_list.append(uflux)

        logger.debug(f'{pflux_list=}')
    
        ##Optional visualization code commented out
        #hp.gnomview(self.m, rot=[self.lon,self.lat,0])
        #for lon, lat in zip(lon_list, lat_list):
        #    hp.projscatter(lon, lat, lonlat=True)
        #plt.show()
    
        # return tuple(pflux_list + lon_list + lat_list)
        pflux_arr = np.array(pflux_list)
        qflux_arr = np.array(qflux_list)
        uflux_arr = np.array(uflux_list)
        ang_near_arr = np.array(ang_near)[0:num_ps]
        lon_arr = np.array(lon_list)
        lat_arr = np.array(lat_list)
        num_ps = np.count_nonzero(np.where(pflux_arr > self.flux2norm_beam(flux=1) / self.nside2pixarea_factor, pflux_arr, 0))
        logger.debug(f'there are {num_ps} ps > 1 mJy')
        logger.debug(f'ang_near_arr before mask very faint: {ang_near_arr}')
        logger.debug(f'lon_arr before mask very faint: {lon_arr}')
        logger.debug(f'lat_arr before mask very faint: {lat_arr}')
        logger.debug(f'pflux_arr before mask very faint: {pflux_arr}')

        mask_very_faint = pflux_arr > self.flux2norm_beam(flux=1) / self.nside2pixarea_factor

        ang_near_arr = ang_near_arr[mask_very_faint].copy()
        pflux_arr = pflux_arr[mask_very_faint].copy()
        qflux_arr = qflux_arr[mask_very_faint].copy()
        uflux_arr = uflux_arr[mask_very_faint].copy()
        lon_arr = lon_arr[mask_very_faint].copy()
        lat_arr = lat_arr[mask_very_faint].copy()

        self.ang_near = ang_near_arr

        logger.debug(f'ang_near_arr after mask very faint: {ang_near_arr}')
        logger.debug(f'lon_arr after mask very faint: {lon_arr}')
        logger.debug(f'lat_arr after mask very faint: {lat_arr}')
        logger.debug(f'pflux_arr after mask very faint: {pflux_arr}')

        if num_ps > 0:
            ang_near_and_bigger_than_threshold = ang_near[0:num_ps]
            if any(ang_near_and_bigger_than_threshold < 0.35):
                self.flag_too_near = True

                self.num_near_ps = np.count_nonzero(np.where(ang_near_and_bigger_than_threshold < 0.35, ang_near_and_bigger_than_threshold, 0))
                logger.debug(f'{self.num_near_ps=}')
                sorted_indices = np.argsort(ang_near_arr)

                ang_near_arr = ang_near_arr[sorted_indices]
                pflux_arr = pflux_arr[sorted_indices]
                qflux_arr = qflux_arr[sorted_indices]
                uflux_arr = uflux_arr[sorted_indices]
                lon_arr = lon_arr[sorted_indices]
                lat_arr = lat_arr[sorted_indices]

                logger.debug(f'ang_near_arr after sort by ang: {ang_near_arr}')
                logger.debug(f'lon_arr after sort by ang: {lon_arr}')
                logger.debug(f'lat_arr after sort by ang: {lat_arr}')
                logger.debug(f'pflux_arr after sort by ang: {pflux_arr}')
                logger.debug(f'qflux_arr after sort by ang: {qflux_arr}')
                logger.debug(f'uflux_arr after sort by ang: {uflux_arr}')

            logger.debug(f'{self.flag_too_near = }')

        return num_ps, tuple(sum(zip(qflux_arr, uflux_arr, lon_arr, lat_arr), ()))

    def fit_all(self, cov_mode:str, mode:str='pipeline'):
        def calc_error():
            theta = hp.rotator.angdist(dir1=ctr0_vec, dir2=vec_around)

            def model():
                return 1 / (2 * np.pi * self.sigma**2) * np.exp(- (theta)**2 / (2 * self.sigma**2))

            y_model = model()
            Fish_mat = y_model @ self.inv_cov @ y_model
            sigma = 1 / np.sqrt(Fish_mat)
            logger.info(f'{sigma=}')

        def lsq_4_params(q_amp,u_amp, c_q, c_u):

            theta = hp.rotator.angdist(dir1=ctr0_vec, dir2=vec_around)

            def model():
                profile = 1 / (2 * np.pi * self.sigma**2) * np.exp(- (theta)**2 / (2 * self.sigma**2))
                P = (q_amp + 1j * u_amp) * np.exp(2j * self.lon_rad)
                lugwid_P = P * profile
                QU = lugwid_P * np.exp(-2j * self.phi_around)
                Q = QU.real + c_q
                U = QU.imag + c_u
                return np.concatenate([Q,U])

            y_model = model()
            y_data = np.concatenate([self.m_q[ipix_fit], self.m_u[ipix_fit]])
            y_err = np.concatenate([self.nstd_q[ipix_fit], self.nstd_u[ipix_fit]])
            y_diff = y_data - y_model

            # error_estimate = np.sum(y_model**2 / y_err**2)
            # print(f"{error_estimate=}")

            z = (y_diff) @ self.inv_cov @ (y_diff)
            return z
            # z = (y_diff) / y_err
            # return np.sum(z**2)

        def lsq_params(*args):
            # args is expected to be in the format:
            # norm_beam1, norm_beamN, const
        
            num_ps = (len(args) - 2) // 2 # Determine the number of point sources based on the number of arguments
        
            # Extract const
            c_u = args[-1]
            c_q = args[-2]
        
            # Process each point source
            thetas = []
            for i in range(num_ps):
                q_amp, u_amp = args[i*2:i*2+2]
                lon = self.fit_lon[i]
                lat = self.fit_lat[i]
                if np.isnan(lon): lon = self.fit_lon[i]+np.random.uniform(-0.01, 0.01)
                if np.isnan(lat): lat = self.fit_lat[i]+np.random.uniform(-0.01, 0.01)
                # print(f'{lon=},{lat=}')
                lat = self.adjust_lat(lat)
                ctr_vec = np.array(hp.ang2vec(theta=lon, phi=lat, lonlat=True))

                theta = hp.rotator.angdist(dir1=ctr_vec, dir2=vec_around)

                profile = 1 / (2 * np.pi * self.sigma**2) * np.exp(- (theta)**2 / (2 * self.sigma**2))
                P = (q_amp + 1j * u_amp) * np.exp(2j * np.deg2rad(lon)) * self.nside2pixarea_factor
                lugwid_P = P * profile
                QU = lugwid_P * np.exp(-2j * self.phi_around)
                Q = QU.real + c_q
                U = QU.imag + c_u
                model = np.concatenate([Q,U])

                thetas.append(model)
        
            def model():
                md = sum(thetas)
                md[:len(Q)] = md[:len(Q)] + c_q
                md[len(Q)+1:-1] = md[len(Q)+1:-1] + c_u
                return md
        
            y_model = model()
            y_data = np.concatenate([self.m_q[ipix_fit], self.m_u[ipix_fit]])
        
            y_diff = y_data - y_model

            z = (y_diff) @ self.inv_cov @ (y_diff)
            logger.debug(f'{z=}')
            return z

        def test_fit():
            params = (self.q_amp, self.u_amp, 0, 0)
            obj_minuit = Minuit(lsq_4_params, name=("q_amp", "u_amp", "c_q", "c_u"), *params)
            obj_minuit.limits = [(-10,10),(-10,10),(-100,100),(-100,100)]
            logger.debug(f'\n{obj_minuit.migrad()}')
            logger.debug(f'\n{obj_minuit.hesse()}')

            chi2dof = obj_minuit.fval / (self.ndof)
            str_chi2 = f"𝜒²/ndof = {obj_minuit.fval:.2f} / {self.ndof} = {chi2dof}"
            logger.debug(str_chi2)
            if obj_minuit.fmin.hesse_failed:
                raise ValueError('hesse failed!')

            logger.info(f'2 parameter fitting is enough, hesse ok')
            return chi2dof, obj_minuit.values['q_amp'],obj_minuit.errors['q_amp'], obj_minuit.values['u_amp'], obj_minuit.errors['u_amp']

        def fit_1_ps():
            params = (self.q_amp, self.u_amp, 0.0, 0.0)
            self.fit_lon = (self.lon,)
            self.fit_lat = (self.lat,)
            logger.debug(f'{self.fit_lon=}, {self.fit_lat=}')

            obj_minuit = Minuit(lsq_params, name=("q_amp_1","u_amp_1","c_q", "c_u"), *params)
            obj_minuit.limits = [(-50000,50000),(-50000,50000), (-500,500), (-500,500)]
            logger.debug(f'\n{obj_minuit.migrad()}')
            logger.debug(f'\n{obj_minuit.hesse()}')

            chi2dof = obj_minuit.fval / self.ndof
            str_chi2 = f"𝜒²/ndof = {obj_minuit.fval:.2f} / {self.ndof} = {chi2dof}"
            logger.debug(str_chi2)

            if obj_minuit.fmin.hesse_failed:
                raise ValueError('hesse failed!')

            logger.info(f'one ps fitting is enough, hesse ok')
            return chi2dof, obj_minuit.values['q_amp_1'],obj_minuit.errors['q_amp_1'], obj_minuit.values['u_amp_1'],obj_minuit.errors['u_amp_1']

        def fit_2_ps():
            num_ps, (self.q_amp_2, self.u_amp_2, self.ctr2_lon, self.ctr2_lat) = self.find_nearby_ps(num_ps=1)
            params = (self.q_amp, self.u_amp, self.q_amp_2, self.u_amp_2, 0.0, 0.0)
            self.fit_lon = (self.lon, self.ctr2_lon)
            self.fit_lat = (self.lat, self.ctr2_lat)
            logger.debug(f'{self.fit_lon=}, {self.fit_lat=}')

            obj_minuit = Minuit(lsq_params, name=("q_amp_1","u_amp_1","q_amp_2","u_amp_2","c_q","c_u"), *params)
            obj_minuit.limits = [(-50000,50000),(-50000,50000),(-50000,50000),(-50000,50000),(-500,500), (-500,500)]
            logger.debug(f'\n{obj_minuit.migrad()}')
            logger.debug(f'\n{obj_minuit.hesse()}')

            chi2dof = obj_minuit.fval / self.ndof
            str_chi2 = f"𝜒²/ndof = {obj_minuit.fval:.2f} / {self.ndof} = {chi2dof}"
            logger.debug(str_chi2)

            if obj_minuit.fmin.hesse_failed:
                raise ValueError('hesse failed!')

            logger.info(f'two ps fitting is enough, hesse ok')
            return chi2dof, obj_minuit.values['q_amp_1'],obj_minuit.errors['q_amp_1'],obj_minuit.values['u_amp_1'],obj_minuit.errors['u_amp_1']

        def fit_3_ps():
            num_ps, (self.q_amp_2, self.u_amp_2, self.ctr2_lon, self.ctr2_lat, self.q_amp_3, self.u_amp_3, self.ctr3_lon, self.ctr3_lat) = self.find_nearby_ps(num_ps=2)
            params = (self.q_amp, self.u_amp, self.q_amp_2, self.u_amp_2, self.q_amp_3, self.u_amp_3, 0.0, 0.0)
            self.fit_lon = (self.lon, self.ctr2_lon, self.ctr3_lon)
            self.fit_lat = (self.lat, self.ctr2_lat, self.ctr3_lat)
            logger.debug(f'{self.fit_lon=}, {self.fit_lat=}')

            obj_minuit = Minuit(lsq_params, name=("q_amp_1","u_amp_1","q_amp_2","u_amp_2","q_amp_3","u_amp_3","c_q","c_u"), *params)
            obj_minuit.limits = [(-50000,50000),(-50000,50000),(-50000,50000),(-50000,50000),(-50000,50000),(-50000,50000),(-500,500), (-500,500)]
            logger.debug(f'\n{obj_minuit.migrad()}')
            logger.debug(f'\n{obj_minuit.hesse()}')

            chi2dof = obj_minuit.fval / self.ndof
            str_chi2 = f"𝜒²/ndof = {obj_minuit.fval:.2f} / {self.ndof} = {chi2dof}"
            logger.debug(str_chi2)

            if obj_minuit.fmin.hesse_failed:
                raise ValueError('hesse failed!')

            logger.info(f'three ps fitting is enough, hesse ok')
            return chi2dof, obj_minuit.values['q_amp_1'],obj_minuit.errors['q_amp_1'],obj_minuit.values['u_amp_1'],obj_minuit.errors['u_amp_1']

        def fit_4_ps():
            num_ps, (self.q_amp_2, self.u_amp_2, self.ctr2_lon, self.ctr2_lat, self.q_amp_3, self.u_amp_3, self.ctr3_lon, self.ctr3_lat, self.q_amp_4, self.u_amp_4, self.ctr4_lon, self.ctr4_lat) = self.find_nearby_ps(num_ps=3)
            params = (self.q_amp, self.u_amp, self.q_amp_2, self.u_amp_2, self.q_amp_3, self.u_amp_3, self.q_amp_4, self.u_amp_4, 0.0, 0.0)
            self.fit_lon = (self.lon, self.ctr2_lon, self.ctr3_lon, self.ctr4_lon)
            self.fit_lat = (self.lat, self.ctr2_lat, self.ctr3_lat, self.ctr4_lat)
            logger.debug(f'{self.fit_lon=}, {self.fit_lat=}')

            obj_minuit = Minuit(lsq_params, name=("q_amp_1","u_amp_1","q_amp_2","u_amp_2","q_amp_3","u_amp_3","q_amp_4","u_amp_4","c_q","c_u"), *params)
            obj_minuit.limits = [(-50000,50000),(-50000,50000),(-50000,50000),(-50000,50000),(-50000,50000),(-50000,50000),(-50000,50000),(-50000,50000),(-500,500), (-500,500)]
            logger.debug(f'\n{obj_minuit.migrad()}')
            logger.debug(f'\n{obj_minuit.hesse()}')

            chi2dof = obj_minuit.fval / self.ndof
            str_chi2 = f"𝜒²/ndof = {obj_minuit.fval:.2f} / {self.ndof} = {chi2dof}"
            logger.debug(str_chi2)

            if obj_minuit.fmin.hesse_failed:
                raise ValueError('hesse failed!')

            logger.info(f'four ps fitting is enough, hesse ok')
            return chi2dof, obj_minuit.values['q_amp_1'],obj_minuit.errors['q_amp_1'],obj_minuit.values['u_amp_1'],obj_minuit.errors['u_amp_1']

        def fit_5_ps():
            num_ps, (self.q_amp_2, self.u_amp_2, self.ctr2_lon, self.ctr2_lat, self.q_amp_3, self.u_amp_3, self.ctr3_lon, self.ctr3_lat, self.q_amp_4, self.u_amp_4, self.ctr4_lon, self.ctr4_lat, self.q_amp_5, self.u_amp_5, self.ctr5_lon, self.ctr5_lat) = self.find_nearby_ps(num_ps=4)
            params = (self.q_amp, self.u_amp, self.q_amp_2, self.u_amp_2, self.q_amp_3, self.u_amp_3, self.q_amp_4, self.u_amp_4, self.q_amp_5, self.u_amp_5, 0.0, 0.0)
            self.fit_lon = (self.lon, self.ctr2_lon, self.ctr3_lon, self.ctr4_lon, self.ctr5_lon)
            self.fit_lat = (self.lat, self.ctr2_lat, self.ctr3_lat, self.ctr4_lat, self.ctr5_lat)
            logger.debug(f'{self.fit_lon=}, {self.fit_lat=}')

            obj_minuit = Minuit(lsq_params, name=("q_amp_1","u_amp_1","q_amp_2","u_amp_2","q_amp_3","u_amp_3","q_amp_4","u_amp_4","q_amp_5","u_amp_5","c_q","c_u"), *params)
            obj_minuit.limits = [(-50000,50000),(-50000,50000),(-50000,50000),(-50000,50000),(-50000,50000),(-50000,50000),(-50000,50000),(-50000,50000),(-50000,50000),(-50000,50000),(-500,500), (-500,500)]
            logger.debug(f'\n{obj_minuit.migrad()}')
            logger.debug(f'\n{obj_minuit.hesse()}')

            chi2dof = obj_minuit.fval / self.ndof
            str_chi2 = f"𝜒²/ndof = {obj_minuit.fval:.2f} / {self.ndof} = {chi2dof}"
            logger.debug(str_chi2)

            if obj_minuit.fmin.hesse_failed:
                raise ValueError('hesse failed!')

            logger.info(f'five ps fitting is enough, hesse ok')
            return chi2dof, obj_minuit.values['q_amp_1'],obj_minuit.errors['q_amp_1'],obj_minuit.values['u_amp_1'],obj_minuit.errors['u_amp_1']

        def fit_6_ps():
            num_ps, (self.q_amp_2, self.u_amp_2, self.ctr2_lon, self.ctr2_lat, self.q_amp_3, self.u_amp_3, self.ctr3_lon, self.ctr3_lat, self.q_amp_4, self.u_amp_4, self.ctr4_lon, self.ctr4_lat, self.q_amp_5, self.u_amp_5, self.ctr5_lon, self.ctr5_lat, self.q_amp_6, self.u_amp_6, self.ctr6_lon, self.ctr6_lat) = self.find_nearby_ps(num_ps=5)
            params = (self.q_amp, self.u_amp, self.q_amp_2, self.u_amp_2, self.q_amp_3, self.u_amp_3, self.q_amp_4, self.u_amp_4, self.q_amp_5, self.u_amp_5, self.q_amp_6, self.u_amp_6, 0.0, 0.0)
            self.fit_lon = (self.lon, self.ctr2_lon, self.ctr3_lon, self.ctr4_lon, self.ctr5_lon, self.ctr6_lon)
            self.fit_lat = (self.lat, self.ctr2_lat, self.ctr3_lat, self.ctr4_lat, self.ctr5_lat, self.ctr6_lat)
            logger.debug(f'{self.fit_lon=}, {self.fit_lat=}')

            obj_minuit = Minuit(lsq_params, name=("q_amp_1","u_amp_1","q_amp_2","u_amp_2","q_amp_3","u_amp_3","q_amp_4","u_amp_4","q_amp_5","u_amp_5","q_amp_6","u_amp_6","c_q","c_u"), *params)
            obj_minuit.limits = [(-50000,50000),(-50000,50000),(-50000,50000),(-50000,50000),(-50000,50000),(-50000,50000),(-50000,50000),(-50000,50000),(-50000,50000),(-50000,50000),(-50000,50000),(-50000,50000),(-500,500), (-500,500)]
            logger.debug(f'\n{obj_minuit.migrad()}')
            logger.debug(f'\n{obj_minuit.hesse()}')

            chi2dof = obj_minuit.fval / self.ndof
            str_chi2 = f"𝜒²/ndof = {obj_minuit.fval:.2f} / {self.ndof} = {chi2dof}"
            logger.debug(str_chi2)

            if obj_minuit.fmin.hesse_failed:
                raise ValueError('hesse failed!')

            logger.info(f'six ps fitting is enough, hesse ok')
            return chi2dof, obj_minuit.values['q_amp_1'],obj_minuit.errors['q_amp_1'],obj_minuit.values['u_amp_1'],obj_minuit.errors['u_amp_1']

        def fit_7_ps():
            num_ps, (self.q_amp_2, self.u_amp_2, self.ctr2_lon, self.ctr2_lat, self.q_amp_3, self.u_amp_3, self.ctr3_lon, self.ctr3_lat, self.q_amp_4, self.u_amp_4, self.ctr4_lon, self.ctr4_lat, self.q_amp_5, self.u_amp_5, self.ctr5_lon, self.ctr5_lat, self.q_amp_6, self.u_amp_6, self.ctr6_lon, self.ctr6_lat, self.q_amp_7, self.u_amp_7, self.ctr7_lon, self.ctr7_lat) = self.find_nearby_ps(num_ps=6)
            params = (self.q_amp, self.u_amp, self.q_amp_2, self.u_amp_2, self.q_amp_3, self.u_amp_3, self.q_amp_4, self.u_amp_4, self.q_amp_5, self.u_amp_5, self.q_amp_6, self.u_amp_6, self.q_amp_7, self.u_amp_7, 0.0, 0.0)
            self.fit_lon = (self.lon, self.ctr2_lon, self.ctr3_lon, self.ctr4_lon, self.ctr5_lon, self.ctr6_lon, self.ctr7_lon)
            self.fit_lat = (self.lat, self.ctr2_lat, self.ctr3_lat, self.ctr4_lat, self.ctr5_lat, self.ctr6_lat, self.ctr7_lat)
            logger.debug(f'{self.fit_lon=}, {self.fit_lat=}')

            obj_minuit = Minuit(lsq_params, name=("q_amp_1","u_amp_1","q_amp_2","u_amp_2","q_amp_3","u_amp_3","q_amp_4","u_amp_4","q_amp_5","u_amp_5","q_amp_6","u_amp_6","q_amp_7","u_amp_7","c_q","c_u"), *params)
            obj_minuit.limits = [(-50000,50000),(-50000,50000),(-50000,50000),(-50000,50000),(-50000,50000),(-50000,50000),(-50000,50000),(-50000,50000),(-50000,50000),(-50000,50000),(-50000,50000),(-50000,50000),(-50000,50000),(-50000,50000),(-500,500), (-500,500)]
            logger.debug(f'\n{obj_minuit.migrad()}')
            logger.debug(f'\n{obj_minuit.hesse()}')

            chi2dof = obj_minuit.fval / self.ndof
            str_chi2 = f"𝜒²/ndof = {obj_minuit.fval:.2f} / {self.ndof} = {chi2dof}"
            logger.debug(str_chi2)

            if obj_minuit.fmin.hesse_failed:
                raise ValueError('hesse failed!')

            logger.info(f'seven ps fitting is enough, hesse ok')
            return chi2dof, obj_minuit.values['q_amp_1'],obj_minuit.errors['q_amp_1'],obj_minuit.values['u_amp_1'],obj_minuit.errors['u_amp_1']

        def fit_8_ps():
            num_ps, (self.q_amp_2, self.u_amp_2, self.ctr2_lon, self.ctr2_lat, self.q_amp_3, self.u_amp_3, self.ctr3_lon, self.ctr3_lat, self.q_amp_4, self.u_amp_4, self.ctr4_lon, self.ctr4_lat, self.q_amp_5, self.u_amp_5, self.ctr5_lon, self.ctr5_lat, self.q_amp_6, self.u_amp_6, self.ctr6_lon, self.ctr6_lat, self.q_amp_7, self.u_amp_7, self.ctr7_lon, self.ctr7_lat, self.q_amp_8, self.u_amp_8, self.ctr8_lon, self.ctr8_lat) = self.find_nearby_ps(num_ps=7)
            params = (self.q_amp, self.u_amp, self.q_amp_2, self.u_amp_2, self.q_amp_3, self.u_amp_3, self.q_amp_4, self.u_amp_4, self.q_amp_5, self.u_amp_5, self.q_amp_6, self.u_amp_6, self.q_amp_7, self.u_amp_7, self.q_amp_8, self.u_amp_8, 0.0, 0.0)
            self.fit_lon = (self.lon, self.ctr2_lon, self.ctr3_lon, self.ctr4_lon, self.ctr5_lon, self.ctr6_lon, self.ctr7_lon, self.ctr8_lon)
            self.fit_lat = (self.lat, self.ctr2_lat, self.ctr3_lat, self.ctr4_lat, self.ctr5_lat, self.ctr6_lat, self.ctr7_lat, self.ctr8_lat)
            logger.debug(f'{self.fit_lon=}, {self.fit_lat=}')

            obj_minuit = Minuit(lsq_params, name=("q_amp_1","u_amp_1","q_amp_2","u_amp_2","q_amp_3","u_amp_3","q_amp_4","u_amp_4","q_amp_5","u_amp_5","q_amp_6","u_amp_6","q_amp_7","u_amp_7","q_amp_8","u_amp_8","c_q","c_u"), *params)
            obj_minuit.limits = [(-50000,50000),(-50000,50000),(-50000,50000),(-50000,50000),(-50000,50000),(-50000,50000),(-50000,50000),(-50000,50000),(-50000,50000),(-50000,50000),(-50000,50000),(-50000,50000),(-50000,50000),(-50000,50000),(-50000,50000),(-50000,50000),(-500,500), (-500,500)]
            logger.debug(f'\n{obj_minuit.migrad()}')
            logger.debug(f'\n{obj_minuit.hesse()}')

            chi2dof = obj_minuit.fval / self.ndof
            str_chi2 = f"𝜒²/ndof = {obj_minuit.fval:.2f} / {self.ndof} = {chi2dof}"
            logger.debug(str_chi2)

            if obj_minuit.fmin.hesse_failed:
                raise ValueError('hesse failed!')

            logger.info(f'eight ps fitting is enough, hesse ok')
            return chi2dof, obj_minuit.values['q_amp_1'],obj_minuit.errors['q_amp_1'],obj_minuit.values['u_amp_1'],obj_minuit.errors['u_amp_1']

        if mode == 'pipeline':
            self.inv_cov = np.load(f'./inv_cov_{self.nside}/r_{self.radius_factor}/{cov_mode}/{self.flux_idx}.npy')

            ctr0_vec = self.ctr0_vec
            ipix_fit = self.ipix_fit
            vec_around = self.vec_around

            num_ps, near = self.find_nearby_ps(num_ps=10)

            logger.info(f'Ready for fitting...')
            logger.info(f'{num_ps=}, {near=}')

            logger.info(f'begin point source fitting, first do one ps fitting...')
            # chi2dof, fit_q_amp, fit_q_amp_err, fit_u_amp, fit_u_amp_err = test_fit()
            chi2dof, fit_q_amp, fit_q_amp_err, fit_u_amp, fit_u_amp_err = fit_1_ps()
            if np.abs(fit_q_amp) < self.sigma_threshold * fit_q_amp_err:
                logger.info('there is no point sources on Q map')
            if np.abs(fit_u_amp) < self.sigma_threshold * fit_u_amp_err:
                logger.info('there is no point sources on U map')

            # if num_ps == 0:
            #     chi2dof, fit_q_amp, fit_q_amp_err, fit_u_amp, fit_u_amp_err = fit_1_ps()
            # elif num_ps == 1:
            #     chi2dof, fit_q_amp, fit_q_amp_err, fit_u_amp, fit_u_amp_err = fit_2_ps()
            # elif num_ps == 2:
            #     chi2dof, fit_q_amp, fit_q_amp_err, fit_u_amp, fit_u_amp_err = fit_3_ps()
            # elif num_ps == 3:
            #     chi2dof, fit_q_amp, fit_q_amp_err, fit_u_amp, fit_u_amp_err = fit_4_ps()
            # elif num_ps == 4:
            #     chi2dof, fit_q_amp, fit_q_amp_err, fit_u_amp, fit_u_amp_err = fit_5_ps()
            # elif num_ps == 5:
            #     chi2dof, fit_q_amp, fit_q_amp_err, fit_u_amp, fit_u_amp_err = fit_6_ps()
            # elif num_ps == 6:
            #     chi2dof, fit_q_amp, fit_q_amp_err, fit_u_amp, fit_u_amp_err = fit_7_ps()
            # elif num_ps == 7:
            #     chi2dof, fit_q_amp, fit_q_amp_err, fit_u_amp, fit_u_amp_err = fit_8_ps()

            fit_P, fit_P_err = FitPolPS.calculate_P_error(Q=fit_q_amp, U=fit_u_amp, sigma_Q=fit_q_amp_err, sigma_U=fit_u_amp_err)
            fit_phi, fit_phi_err = FitPolPS.calculate_phi_error(Q=fit_q_amp, U=fit_u_amp, sigma_Q=fit_q_amp_err, sigma_U=fit_u_amp_err)

            fit_error_q = np.abs((fit_q_amp - self.q_amp) / self.q_amp )
            fit_error_u = np.abs((fit_u_amp - self.u_amp) / self.u_amp )

            logger.info(f'{num_ps=}, {chi2dof=}')
            logger.info(f'{self.q_amp=}, {fit_q_amp=}, {fit_error_q=}, {fit_q_amp_err=}')
            logger.info(f'{self.u_amp=}, {fit_u_amp=}, {fit_error_u=}, {fit_u_amp_err=}')
            logger.info(f'{self.p_amp=}, {fit_P=}, {fit_P_err=}')
            logger.info(f'{self.phi=}, {fit_phi=}, {fit_phi_err=}')
            # return num_ps, chi2dof, fit_q_amp, fit_q_amp_err, fit_u_amp, fit_u_amp_err, fit_error_q, fit_error_u
            return num_ps, chi2dof, fit_P, fit_P_err, fit_phi, fit_phi_err

        if mode == 'get_num_ps':
            num_ps, near = self.find_nearby_ps(num_ps=10)
            return num_ps, self.num_near_ps, self.ang_near

        if mode == 'check_sigma':
            calc_error()

        if mode == 'get_overlap':
            num_ps, near = self.find_nearby_ps(num_ps=10)
            return self.flag_overlap


def main():
    freq = 30
    lmax = 1999
    nside = 2048
    npix = hp.nside2npix(nside)
    beam = 67

    time0 = time.perf_counter()
    nstd = np.load(f'../../FGSim/NSTDNORTH/2048/{freq}.npy')
    print(f'{nstd[1,0]=}')
    nstd_q = nstd[1].copy()
    nstd_u = nstd[2].copy()
    ps = np.load('./data/ps/ps.npy')
    # noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))
    # m = ps + noise
    m = np.load('./data/pcn.npy')
    # m = np.load(f'../../fitdata/synthesis_data/2048/PSCMBNOISE/')
    # m = np.load(f'../../fitdata/synthesis_data/2048/PSCMBNOISE/{freq}/3.npy')
    # m = np.load(f'./1_6k_pcn.npy')
    m_q = m[1].copy()
    m_u = m[2].copy()
    logger.debug(f'{sys.getrefcount(m_q)-1=}')

    logger.info(f'time for fitting = {time.perf_counter()-time0}')

    df_mask = pd.read_csv('../../pp_P/mask/mask_csv/30.csv')
    df_ps = pd.read_csv('../../pp_P/mask/ps_csv/30.csv')

    flux_idx = 0

    logger.debug(f'{sys.getrefcount(m_q)-1=}')
    obj = FitPolPS(m_q=m_q, m_u=m_u, freq=freq, nstd_q=nstd_q, nstd_u=nstd_u, flux_idx=flux_idx, df_mask=df_mask, df_ps=df_ps, lmax=lmax, nside=nside, radius_factor=1.5, beam=beam, epsilon=0.00001)

    logger.debug(f'{sys.getrefcount(m_q)-1=}')
    # obj.see_true_map(m_q=m_q, m_u=m_u, nside=nside, beam=beam)

    # obj.calc_definite_fixed_cmb_cov()
    # obj.calc_covariance_matrix(mode='cmb+noise')
    obj.fit_all(cov_mode='cmb+noise')

    # obj.calc_covariance_matrix(mode='noise')
    # obj.fit_all(cov_mode='noise')

    # obj.fit_all(cov_mode='noise', mode='check_sigma')


if __name__ == '__main__':
    main()





