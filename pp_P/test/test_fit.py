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

    def __init__(self, m_q, m_u, freq, nstd_q, nstd_u, flux_idx, df_mask, df_ps, cl_cmb, lmax, nside, radius_factor, beam, sigma_threshold=5, epsilon=1e-4, debug_flag=False):
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
        self.cl_cmb = cl_cmb # power spectrum of CMB
        self.lmax = lmax # maximum multipole
        self.nside = nside # resolution of healpy maps
        self.radius_factor = radius_factor # disc radius of fitting region
        self.sigma_threshold = sigma_threshold # judge if a signal is a point source
        self.beam = beam # in arcmin
        self.epsilon = epsilon # if CMB covariance matrix is not semi-positive, add this to cross term

        self.i_amp = self.flux2norm_beam(self.iflux)
        self.q_amp = self.flux2norm_beam(self.qflux)
        self.u_amp = self.flux2norm_beam(self.uflux)

        self.sigma = np.deg2rad(beam) / 60 / (np.sqrt(8 * np.log(2)))

        ctr0_pix = hp.ang2pix(nside=self.nside, theta=self.lon, phi=self.lat, lonlat=True)
        self.ctr0_vec = np.array(hp.pix2vec(nside=self.nside, ipix=ctr0_pix)).astype(np.float64)

        self.ipix_fit = hp.query_disc(nside=self.nside, vec=self.ctr0_vec, radius=self.radius_factor * np.deg2rad(self.beam) / 60)

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
        self.ndof = len(self.ipix_fit)

        self.num_near_ps = 0
        self.flag_too_near = False
        self.flag_overlap = False

        logger.info(f'{lmax=}, {nside=}')
        logger.info(f'{freq=}, {beam=}, {flux_idx=}, {radius_factor=}, lon={self.lon}, lat={self.lat}, ndof={self.ndof}')
        logger.info(f'iflux={self.iflux=}, {self.qflux=}, {self.uflux=}')
        logger.info(f'i_amp={self.i_amp}, q_amp={self.q_amp}, u_amp={self.u_amp}')

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

        hp.gnomview(irps, rot=[lon, lat, 0], xsize=300, ysize=300, reso=1, title='irps', sub=223)
        hp.gnomview(radiops, rot=[lon, lat, 0], xsize=300, ysize=300, reso=1, title='radiops', sub=224)
        hp.gnomview(m_q, rot=[lon, lat, 0], xsize=300, ysize=300, sub=221, title='Q map')
        hp.gnomview(m_u, rot=[lon, lat, 0], xsize=300, ysize=300, sub=222, title='U map')
        plt.show()

        vec = hp.ang2vec(theta=lon, phi=lat, lonlat=True)
        ipix_disc = hp.query_disc(nside=nside, vec=vec, radius=np.deg2rad(beam)/60)

        mask = np.ones(hp.nside2npix(nside))
        mask[ipix_disc] = 0

        hp.gnomview(mask, rot=[lon, lat, 0])
        plt.show()

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
        iflux_list = []
        for i in range(min(num_ps, len(index_near[0]))):
            index = index_near[0][i]
            if index < self.df_mask.at[self.flux_idx, 'flux_idx']:
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

        def lsq_2_params(q_amp, const):

            theta = hp.rotator.angdist(dir1=ctr0_vec, dir2=vec_around)

            def model():
                profile = 1 / (2 * np.pi * self.sigma**2) * np.exp(- (theta)**2 / (2 * self.sigma**2))
                P = (q_amp + 1j * 19.3e-3) * np.exp(2j * self.lon_rad)
                lugwid_P = P * profile
                QU = lugwid_P * np.exp(-2j * self.phi_around)
                Q = QU.real
                return Q + const

            y_model = model()
            y_data = self.m_q[ipix_fit]
            y_err = self.nstd_q[ipix_fit]
            y_diff = y_data - y_model

            # error_estimate = np.sum(y_model**2 / y_err**2)
            # print(f"{error_estimate=}")

            # z = (y_diff) @ self.inv_cov @ (y_diff)
            # return z
            z = (y_diff) / y_err
            return np.sum(z**2)

        def lsq_params(*args):
            # args is expected to be in the format:
            # norm_beam1, norm_beamN, const
        
            num_ps = (len(args) - 1)  # Determine the number of point sources based on the number of arguments
        
            # Extract const
            const = args[-1]
        
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
                thetas.append(norm_beam / (2 * np.pi * self.sigma**2) * np.exp(- (theta)**2 / (2 * self.sigma**2)))
        
            def model():
                return sum(thetas) + const
        
            y_model = model()
            y_data = self.m[ipix_fit]
            y_err = self.nstd[ipix_fit]
        
            y_diff = y_data - y_model

            z = (y_diff) @ self.inv_cov @ (y_diff)
            logger.debug(f'{z=}')
            return z

        # self.inv_cov = np.load(f'./inv_cov_{self.nside}/r_{self.radius_factor}/{cov_mode}/{self.flux_idx}.npy')

        ctr0_vec = self.ctr0_vec
        ipix_fit = self.ipix_fit
        vec_around = self.vec_around

        num_ps, near = self.find_nearby_ps(num_ps=10)

        logger.info(f'Ready for fitting...')
        logger.info(f'{num_ps=}, {near=}')

        def test_fit():
            params = (self.q_amp, 0)
            obj_minuit = Minuit(lsq_2_params, name=("q_amp", "const"), *params)
            obj_minuit.limits = [(-1,1), (-100,100)]
            logger.debug(f'\n{obj_minuit.migrad()}')
            logger.debug(f'\n{obj_minuit.hesse()}')

            chi2dof = obj_minuit.fval / (self.ndof)
            str_chi2 = f"𝜒²/ndof = {obj_minuit.fval:.2f} / {self.ndof} = {chi2dof}"
            logger.debug(str_chi2)
            if obj_minuit.fmin.hesse_failed:
                raise ValueError('hesse failed!')

            logger.info(f'2 parameter fitting is enough, hesse ok')
            return chi2dof, obj_minuit.values['q_amp'],obj_minuit.errors['q_amp']


        def fit_2_params():
            params = (self.ini_norm_beam, 0.0)
            self.fit_lon = (self.lon,)
            self.fit_lat = (self.lat,)
            logger.debug(f'{self.fit_lon=}, {self.fit_lat=}')

            obj_minuit = Minuit(lsq_params, name=("norm_beam1","const"), *params)
            obj_minuit.limits = [(-1,1), (-1000,1000)]
            logger.debug(f'\n{obj_minuit.migrad()}')
            logger.debug(f'\n{obj_minuit.hesse()}')

            chi2dof = obj_minuit.fval / self.ndof
            str_chi2 = f"𝜒²/ndof = {obj_minuit.fval:.2f} / {self.ndof} = {chi2dof}"
            logger.debug(str_chi2)

            if obj_minuit.fmin.hesse_failed:
                raise ValueError('hesse failed!')

            logger.info(f'2 parameter fitting is enough, hesse ok')
            return chi2dof, obj_minuit.values['norm_beam1'],obj_minuit.errors['norm_beam1']

        logger.info(f'begin point source fitting, first do 2 parameter fit...')
        chi2dof, fit_q_amp, fit_q_amp_err = test_fit()
        # chi2dof, fit_norm, norm_error = fit_2_params()
        if fit_norm < self.sigma_threshold * norm_error:
            logger.info('there is no point sources.')

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

        if mode == 'get_overlap':
            return self.flag_overlap


def main():
    freq = 270
    time0 = time.perf_counter()
    # m = np.load(f'../../fitdata/synthesis_data/2048/PSNOISE/{freq}/0.npy')[0]
    m_q = np.load(f'../../fitdata/synthesis_data/2048/PSNOISE/{freq}/0.npy')[1].copy()
    m_u = np.load(f'../../fitdata/synthesis_data/2048/PSNOISE/{freq}/0.npy')[2].copy()
    # m = np.load(f'../../fitdata/2048/PS/270/ps.npy')[1]
    logger.debug(f'{sys.getrefcount(m_q)-1=}')


    logger.info(f'time for fitting = {time.perf_counter()-time0}')
    nstd_q = np.load(f'../../FGSim/NSTDNORTH/2048/{freq}.npy')[1].copy()
    nstd_u = np.load(f'../../FGSim/NSTDNORTH/2048/{freq}.npy')[2].copy()
    df_mask = pd.read_csv(f'../mask/mask_csv/{freq}.csv')
    df_ps = pd.read_csv(f'../mask/ps_csv/{freq}.csv')
    lmax = 1999
    nside = 2048
    beam = 9
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

    flux_idx = 1

    logger.debug(f'{sys.getrefcount(m_q)-1=}')
    obj = FitPolPS(m_q=m_q, m_u=m_u, freq=freq, nstd_q=nstd_q, nstd_u=nstd_u, flux_idx=flux_idx, df_mask=df_mask, df_ps=df_ps, cl_cmb=cl_cmb, lmax=lmax, nside=nside, radius_factor=1.5, beam=beam, epsilon=0.00001)

    logger.debug(f'{sys.getrefcount(m_q)-1=}')
    obj.see_true_map(m_q=m_q, m_u=m_u, nside=nside, beam=beam)

    # obj.calc_covariance_matrix(mode='noise', cmb_cov_fold='../cmb_cov_calc/cov')

    # obj.calc_C_theta_itp_func()
    # obj.calc_C_theta(save_path='./cov_r_2.0/2048')
    # obj.calc_precise_C_theta()

    # obj.calc_cmb_cov()
    # obj.calc_definite_fixed_cmb_cov()
    # obj.calc_covariance_matrix(mode='cmb+noise')
    # obj.calc_covariance_matrix(mode='noise')

    obj.fit_all(cov_mode='cmb+noise')
    # obj.fit_all(cov_mode='noise', mode='check_sigma')
    # obj.fit_all(cov_mode='noise')


if __name__ == '__main__':
    main()








