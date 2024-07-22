import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt
import logging

from iminuit import Minuit
from pathlib import Path

# logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s -%(name)s - %(message)s')
logging.basicConfig(level=logging.WARNING)
# logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)

class Fit_on_B:
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

    def __init__(self, m, df_mask, df_ps, flux_idx, qflux, uflux, pflux, lmax, nside, beam, lon, lat, freq, r_fold=3, r_fold_rmv=5):
        self.m = m
        self.df_mask = df_mask
        self.df_ps = df_ps
        self.flux_idx = flux_idx
        self.qflux = qflux # in mJy
        self.uflux = uflux # in mJy
        self.pflux = pflux # in mJy
        self.lmax = lmax
        self.nside = nside
        self.beam = beam
        self.lon = np.rad2deg(lon) # in degree
        self.lat = np.rad2deg(lat) # in degree
        self.freq = freq # in GHz
        self.r_fold = r_fold
        self.r_fold_rmv = r_fold_rmv

        self.nside2pixarea_factor = hp.nside2pixarea(nside=self.nside)
        self.Q = self.flux2norm_beam(qflux) / self.nside2pixarea_factor
        self.U = self.flux2norm_beam(uflux) / self.nside2pixarea_factor
        self.P = self.flux2norm_beam(pflux) / self.nside2pixarea_factor
        self.ps_2phi = np.arctan2(self.U, self.Q)
        logger.info(f'Q={self.Q}, U={self.U}, P={self.P}, phi={self.ps_2phi}')

        self.npix = hp.nside2npix(nside)
        self.sigma = np.deg2rad(beam) / 60 / (np.sqrt(8 * np.log(2)))

        ipix_ctr = hp.ang2pix(theta=self.lon, phi=self.lat, lonlat=True, nside=nside)
        self.pix_lon, self.pix_lat = hp.pix2ang(ipix=ipix_ctr, nside=nside, lonlat=True) # lon lat in degree
        self.ctr_vec = np.array(hp.pix2vec(nside=nside, ipix=ipix_ctr))

        ctr_theta, ctr_phi = hp.pix2ang(nside=nside, ipix=ipix_ctr) # center pixel theta phi in sphere coordinate
        self.vec_theta = np.asarray((np.cos(ctr_theta)*np.cos(ctr_phi), np.cos(ctr_theta)*np.sin(ctr_phi), -np.sin(ctr_theta)))
        self.vec_phi = np.asarray((-np.sin(ctr_phi), np.cos(ctr_phi), 0))

        self.num_near_ps = 0
        self.flag_too_near = False
        self.flag_overlap = False


    # Utils
    def flux2norm_beam(self, flux):
        # from mJy to muK_CMB to norm_beam
        coeffmJy2norm = Fit_on_B.mJy_to_uKCMB(1, self.freq)
        # logger.debug(f'{coeffmJy2norm=}')
        return coeffmJy2norm * flux

    # Set up parameters
    def params_for_fitting(self):
        self.ipix_disc = hp.query_disc(nside=self.nside, vec=self.ctr_vec, radius=self.r_fold * np.deg2rad(self.beam) / 60 ) # disc for fitting
        path_pix_idx = Path('./pix_idx')
        path_pix_idx.mkdir(exist_ok=True, parents=True)
        np.save(f'./pix_idx/{self.flux_idx}.npy', self.ipix_disc)
        self.ndof = np.size(self.ipix_disc) # degree of freedom
        logger.debug(f'{self.ipix_disc.shape=}, {self.ndof=}')

        self.vec_disc = np.array(hp.pix2vec(nside=self.nside, ipix=self.ipix_disc.astype(int))).astype(np.float64)
        vec_ctr_to_disc = self.vec_disc.T - self.ctr_vec # vector from center to fitting point

        r = np.linalg.norm(vec_ctr_to_disc, axis=1) # radius in polar coordinate
        # np.set_printoptions(threshold=np.inf)
        # print(f'{r=}')

        normed_vec_ctr_to_disc = vec_ctr_to_disc.T / r # normed vector from center to fitting point for calculating xi
        normed_vec_ctr_to_disc = np.nan_to_num(normed_vec_ctr_to_disc, nan=0)
        logger.debug(f'{normed_vec_ctr_to_disc=}')

        cos_theta = normed_vec_ctr_to_disc.T @ self.vec_theta
        cos_phi = normed_vec_ctr_to_disc.T @ self.vec_phi

        xi = np.arctan2(cos_phi, cos_theta) # xi in polar coordinate
        self.cos_2xi = np.cos(2*xi)
        self.sin_2xi = np.sin(2*xi)
        logger.debug(f'{xi=}')

        self.r_2 = r**2
        self.r_2_div_sigma = self.r_2 / (2 * self.sigma**2)

    def params_for_testing(self):
        self.ipix_disc = hp.query_disc(nside=self.nside, vec=self.ctr_vec, radius=self.r_fold_rmv * np.deg2rad(self.beam) / 60 ) # disc for fitting
        self.ndof = np.size(self.ipix_disc) # degree of freedom
        logger.debug(f'{self.ipix_disc.shape=}, {self.ndof=}')

        vec_disc = np.array(hp.pix2vec(nside=self.nside, ipix=self.ipix_disc.astype(int))).astype(np.float64)
        vec_ctr_to_disc = vec_disc.T - self.ctr_vec # vector from center to fitting point

        r = np.linalg.norm(vec_ctr_to_disc, axis=1) # radius in polar coordinate
        # np.set_printoptions(threshold=np.inf)
        # print(f'{r=}')

        normed_vec_ctr_to_disc = vec_ctr_to_disc.T / r # normed vector from center to fitting point for calculating xi
        normed_vec_ctr_to_disc = np.nan_to_num(normed_vec_ctr_to_disc, nan=0)
        logger.debug(f'{normed_vec_ctr_to_disc=}')

        cos_theta = normed_vec_ctr_to_disc.T @ self.vec_theta
        cos_phi = normed_vec_ctr_to_disc.T @ self.vec_phi

        xi = np.arctan2(cos_phi, cos_theta) # xi in polar coordinate
        self.cos_2xi = np.cos(2*xi)
        self.sin_2xi = np.sin(2*xi)
        logger.debug(f'{xi=}')

        self.r_2 = r**2
        self.r_2_div_sigma = self.r_2 / (2 * self.sigma**2)

    # find the point sources and return there parameter
    def find_nearby_ps(self, num_ps=1, threshold_extra_factor=1.1):
        threshold_factor = self.r_fold + threshold_extra_factor
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
        ps_2phi_list = []
        for i in range(min(num_ps, len(index_near[0]))):
            index = index_near[0][i]
            if index < self.df_mask.at[self.flux_idx, 'flux_idx']:
                lon = np.rad2deg(self.df_ps.at[index, 'lon'])
                lat = np.rad2deg(self.df_ps.at[index, 'lat'])
                pflux = self.flux2norm_beam(self.df_ps.at[index, 'pflux']) / self.nside2pixarea_factor
                qflux = self.flux2norm_beam(self.df_ps.at[index, 'qflux']) / self.nside2pixarea_factor
                uflux = self.flux2norm_beam(self.df_ps.at[index, 'uflux']) / self.nside2pixarea_factor
                ps_2phi = np.arctan2(uflux, qflux)
            else:
                lon = np.rad2deg(self.df_ps.at[index + 1, 'lon'])
                lat = np.rad2deg(self.df_ps.at[index + 1, 'lat'])
                pflux = self.flux2norm_beam(self.df_ps.at[index + 1, 'pflux']) / self.nside2pixarea_factor
                qflux = self.flux2norm_beam(self.df_ps.at[index + 1, 'qflux']) / self.nside2pixarea_factor
                uflux = self.flux2norm_beam(self.df_ps.at[index + 1, 'uflux']) / self.nside2pixarea_factor
                ps_2phi = np.arctan2(uflux, qflux)
            lon_list.append(lon)
            lat_list.append(lat)
            pflux_list.append(pflux)
            qflux_list.append(qflux)
            uflux_list.append(uflux)
            ps_2phi_list.append(ps_2phi)

        logger.debug(f'{pflux_list=}')
        logger.debug(f'{ps_2phi_list=}')
    
        ##Optional visualization code commented out
        #hp.gnomview(self.m, rot=[np.rad2deg(self.lon),np.rad2deg(self.lat),0], xsize=1000)
        #for lon, lat in zip(lon_list, lat_list):
        #    hp.projscatter(lon, lat, lonlat=True)
        #plt.show()
    
        # return tuple(pflux_list + lon_list + lat_list)
        pflux_arr = np.array(pflux_list)
        qflux_arr = np.array(qflux_list)
        uflux_arr = np.array(uflux_list)
        ps_2phi_arr = np.array(ps_2phi_list)
        ang_near_arr = np.array(ang_near)[0:num_ps]
        lon_arr = np.array(lon_list)
        lat_arr = np.array(lat_list)
        logger.info(f'{self.flux2norm_beam(flux=1)/self.nside2pixarea_factor=}')
        flux_density_threshold = self.flux2norm_beam(flux=3)/self.nside2pixarea_factor
        num_ps = np.count_nonzero(np.where(pflux_arr > flux_density_threshold, pflux_arr, 0))
        logger.info(f'there are {num_ps} ps > 3 mJy')
        logger.info(f'ang_near_arr before mask very faint: {ang_near_arr}')
        logger.info(f'lon_arr before mask very faint: {lon_arr}')
        logger.info(f'lat_arr before mask very faint: {lat_arr}')
        logger.info(f'pflux_arr before mask very faint: {pflux_arr}')
        logger.info(f'ps_2phi before mask very faint: {ps_2phi_arr}')

        mask_very_faint = pflux_arr > flux_density_threshold

        ang_near_arr = ang_near_arr[mask_very_faint].copy()
        pflux_arr = pflux_arr[mask_very_faint].copy()
        qflux_arr = qflux_arr[mask_very_faint].copy()
        uflux_arr = uflux_arr[mask_very_faint].copy()
        lon_arr = lon_arr[mask_very_faint].copy()
        lat_arr = lat_arr[mask_very_faint].copy()
        ps_2phi_arr = ps_2phi_arr[mask_very_faint].copy()

        self.ang_near = ang_near_arr

        logger.info(f'ang_near_arr after mask very faint: {ang_near_arr}')
        logger.info(f'lon_arr after mask very faint: {lon_arr}')
        logger.info(f'lat_arr after mask very faint: {lat_arr}')
        logger.info(f'pflux_arr after mask very faint: {pflux_arr}')
        logger.info(f'ps_2phi_arr after mask very faint: {ps_2phi_arr}')

        if num_ps > 0:
            ang_near_and_bigger_than_threshold = ang_near[0:num_ps]
            if any(ang_near_and_bigger_than_threshold < 0.35):
                self.flag_too_near = True

                self.num_near_ps = np.count_nonzero(np.where(ang_near_and_bigger_than_threshold < 0.35, ang_near_and_bigger_than_threshold, 0))
                logger.info(f'{self.num_near_ps=}')
                sorted_indices = np.argsort(ang_near_arr)

                ang_near_arr = ang_near_arr[sorted_indices]
                pflux_arr = pflux_arr[sorted_indices]
                qflux_arr = qflux_arr[sorted_indices]
                uflux_arr = uflux_arr[sorted_indices]
                lon_arr = lon_arr[sorted_indices]
                lat_arr = lat_arr[sorted_indices]
                ps_2phi_arr = ps_2phi_arr[sorted_indices]

                logger.info(f'ang_near_arr after sort by ang: {ang_near_arr}')
                logger.info(f'lon_arr after sort by ang: {lon_arr}')
                logger.info(f'lat_arr after sort by ang: {lat_arr}')
                logger.info(f'pflux_arr after sort by ang: {pflux_arr}')
                logger.info(f'qflux_arr after sort by ang: {qflux_arr}')
                logger.info(f'uflux_arr after sort by ang: {uflux_arr}')
                logger.info(f'ps_2phi_arr after sort by ang: {ps_2phi_arr}')

            logger.info(f'{self.flag_too_near = }')

        # return num_ps, tuple(sum(zip(qflux_arr, uflux_arr, lon_arr, lat_arr), ()))
        return num_ps, tuple(sum(zip(pflux_arr, ps_2phi_arr, lon_arr, lat_arr), ()))


    # Calculate inverse covariance matrix
    def calc_inv_cov(self, mode='n'):

        if mode == 'cn1':
            cmb_cov = np.load(f'./cmb_b_cov/{self.flux_idx}.npy') # load cmb cov
            # cmb_cov = np.load('./data/c_cov.npy')

            logger.debug(f'{cmb_cov.shape=}')
            # noise_cov = np.load('./data/noise_cov.npy')
            noise_cov = np.load(f'./noise_b_cov/{self.flux_idx}.npy')
            logger.debug(f'{noise_cov=}')
            cov = cmb_cov + noise_cov
            eigenval, eigenvec = np.linalg.eigh(cov)
            logger.debug(f'{eigenval=}')
            eigenval[eigenval < 0] = 1e-10
            reconstructed_cov = np.dot(eigenvec * eigenval, eigenvec.T)

            self.inv_cov = np.linalg.inv(reconstructed_cov)
            return None

        if mode == 'n1':
            # noise_cov = np.load('./data/noise_cov.npy')
            noise_cov = np.load('./noise_b_cov/0.npy')
            cov = noise_cov
            self.inv_cov = np.linalg.inv(cov)
            return None

        if mode == "c":
            cov = np.load('./cmb_b_cov/0.npy') # load cmb cov

            eigenval, eigenvec = np.linalg.eigh(cov)
            logger.debug(f'{eigenval=}')
            eigenval[eigenval < 0] = 1e-10
            reconstructed_cov = np.dot(eigenvec * eigenval, eigenvec.T)

            self.inv_cov = np.linalg.inv(reconstructed_cov)
            return None

        if mode == 'cn':
            cov = np.load(f'./cmb_b_cov/{self.flux_idx}.npy') # load cmb cov
            # cov = np.load('./data/c_cov.npy')
            logger.debug(f'{cov.shape=}')

            eigenval, eigenvec = np.linalg.eigh(cov)
            logger.debug(f'{eigenval=}')
            eigenval[eigenval < 0] = 1e-10
            cov = np.dot(eigenvec * eigenval, eigenvec.T)

            # nstd = 0.85661

        if mode == 'n':
            cov = np.zeros((self.ndof, self.ndof))

        nstd = 0.1
        # nstd = 0.086
        nstd2 = nstd**2

        for i in range(self.ndof):
            cov[i,i] = cov[i,i] + nstd2

        self.inv_cov = np.linalg.inv(cov)

    # Code for fitting
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

    def model(self, A, ps_2phi):
        model = - A * self.nside2pixarea_factor / (np.pi) * (self.sin_2xi * np.cos(ps_2phi) - self.cos_2xi * np.sin(ps_2phi)) * (1 / self.r_2) * (np.exp(-self.r_2_div_sigma) * (1+self.r_2_div_sigma) - 1)
        model = np.nan_to_num(model, nan=0)
        return model

    def lsq(self, A, ps_2phi, const):
        y_model = self.model(A, ps_2phi) + const
        y_data = self.m[self.ipix_disc]
        y_diff = y_data - y_model

        z = y_diff @ self.inv_cov @ y_diff
        return z

    def lsq_params(self, *args):
        # args is expected to be in the format:
        # Npolarization flux density, Npolarization angle, const

        num_ps = (len(args) - 1) // 2 # Determine the number of point sources based on the number of arguments

        # Extract const
        c = args[-1]

        # Process each point source
        models = []
        for i in range(num_ps):
            A, ps_2phi = args[i*2:i*2+2]
            lon = self.fit_lon[i]
            lat = self.fit_lat[i]

            logger.debug(f'{lon=}, {lat=}')
            # if np.isnan(lon): lon = self.fit_lon[i]+np.random.uniform(-0.01, 0.01)
            # if np.isnan(lat): lat = self.fit_lat[i]+np.random.uniform(-0.01, 0.01)
            # print(f'{lon=},{lat=}')
            # lat = self.adjust_lat(lat)

            ipix_ctr = hp.ang2pix(nside=self.nside, theta=lon, phi=lat, lonlat=True)
            ctr_theta, ctr_phi = hp.pix2ang(nside=self.nside, ipix=ipix_ctr) # center pixel theta phi in sphere coordinate
            vec_theta = np.asarray((np.cos(ctr_theta)*np.cos(ctr_phi), np.cos(ctr_theta)*np.sin(ctr_phi), -np.sin(ctr_theta)))
            vec_phi = np.asarray((-np.sin(ctr_phi), np.cos(ctr_phi), 0))

            ctr_vec = np.array(hp.ang2vec(theta=lon, phi=lat, lonlat=True))
            vec_ctr_to_disc = self.vec_disc.T - ctr_vec # vector from center to fitting point
            r = np.linalg.norm(vec_ctr_to_disc, axis=1) # radius in polar coordinate
            # np.set_printoptions(threshold=np.inf)
            # print(f'{r=}')

            normed_vec_ctr_to_disc = vec_ctr_to_disc.T / r # normed vector from center to fitting point for calculating xi
            normed_vec_ctr_to_disc = np.nan_to_num(normed_vec_ctr_to_disc, nan=0)
            # logger.debug(f'{normed_vec_ctr_to_disc=}')

            cos_theta = normed_vec_ctr_to_disc.T @ vec_theta
            cos_phi = normed_vec_ctr_to_disc.T @ vec_phi

            xi = np.arctan2(cos_phi, cos_theta) # xi in polar coordinate
            cos_2xi = np.cos(2*xi)
            sin_2xi = np.sin(2*xi)
            # logger.debug(f'{xi=}')

            r_2 = r**2
            r_2_div_sigma = r_2 / (2 * self.sigma**2)

            model = - A * self.nside2pixarea_factor / (np.pi) * (sin_2xi * np.cos(ps_2phi) - self.cos_2xi * np.sin(ps_2phi)) * (1 / r_2) * (np.exp(-r_2_div_sigma) * (1+r_2_div_sigma) - 1)
            model = np.nan_to_num(model, nan=0)

            models.append(model)

        y_model = sum(models) + c
        y_data = self.m[self.ipix_disc]

        y_diff = y_data - y_model

        z = (y_diff) @ self.inv_cov @ (y_diff)
        logger.debug(f'{z=}')
        return z

    def fit_1_ps(self):
        num_ps, near = self.find_nearby_ps(num_ps=16, threshold_extra_factor=9.5)
        print(f'{num_ps=}, {near=}')
        params = (self.P, self.ps_2phi, 0.0)
        self.fit_lon = (self.lon,)
        self.fit_lat = (self.lat,)
        logger.debug(f'{self.fit_lon=}, {self.fit_lat=}')

        obj_minuit = Minuit(self.lsq_params, name=("A", "ps_2phi", "const"), *params)
        obj_minuit.limits = [(0,1e5), (-np.pi,np.pi), (-100,100)]
        logger.debug(f'\n{obj_minuit.migrad()}')
        # logger.debug(f'\n{obj_minuit.hesse()}')
        logger.debug(f'\n{obj_minuit.minos()}')

        chi2dof = obj_minuit.fval / self.ndof
        str_chi2 = f"𝜒²/ndof = {obj_minuit.fval:.2f} / {self.ndof} = {chi2dof}"
        logger.info(str_chi2)
        self.A = obj_minuit.values["A"]
        self.A_err = obj_minuit.errors["A"]
        self.P = self.A
        self.P_err = self.A_err
        logger.info(f'P={self.P}')
        logger.info(f'P_err={self.P_err}')
        self.ps_2phi = obj_minuit.values["ps_2phi"]
        self.ps_2phi_err = obj_minuit.errors["ps_2phi"]
        logger.info(f'phi={self.ps_2phi}, phi_err={self.ps_2phi_err}')
        logger.info(f'Q = {self.P*np.cos(self.ps_2phi)}')
        logger.info(f'U = {self.P*np.sin(self.ps_2phi)}')

    def fit_2_ps(self):
        num_ps, (P2, phi2, lon2, lat2) = self.find_nearby_ps(num_ps=1, threshold_extra_factor=9.5)
        print(f'{num_ps=}, {P2=}, {phi2=}, {lon2}, {lat2}')
        params = (self.P, self.ps_2phi, P2, phi2, 0.0)
        # params = (self.P, self.ps_2phi, 100, phi2, 0.0)
        self.fit_lon = (self.lon, lon2)
        self.fit_lat = (self.lat, lat2)
        logger.debug(f'{self.fit_lon=}, {self.fit_lat=}')

        obj_minuit = Minuit(self.lsq_params, name=("A", "ps_2phi", "A2", "phi2", "const"), *params)
        obj_minuit.limits = [(0,1e5), (-np.pi,np.pi), (0,1e5), (-np.pi,np.pi), (-100,100)]
        logger.debug(f'\n{obj_minuit.migrad()}')
        logger.debug(f'\n{obj_minuit.hesse()}')
        # logger.debug(f'\n{obj_minuit.minos()}')

        chi2dof = obj_minuit.fval / self.ndof
        str_chi2 = f"𝜒²/ndof = {obj_minuit.fval:.2f} / {self.ndof} = {chi2dof}"
        logger.info(str_chi2)
        self.A = obj_minuit.values["A"]
        self.A_err = obj_minuit.errors["A"]
        self.P = self.A
        self.P_err = self.A_err
        logger.info(f'P={self.P}')
        logger.info(f'P_err={self.P_err}')
        self.ps_2phi = obj_minuit.values["ps_2phi"]
        self.ps_2phi_err = obj_minuit.errors["ps_2phi"]
        logger.info(f'phi={self.ps_2phi}, phi_err={self.ps_2phi_err}')
        logger.info(f'Q = {self.P*np.cos(self.ps_2phi)}')
        logger.info(f'U = {self.P*np.sin(self.ps_2phi)}')

    def ez_fit_b(self):
        params = (0, 0, 0.0)
        obj_minuit = Minuit(self.lsq, name=("A", "ps_2phi", "const"), *params)
        obj_minuit.limits = [(0,1e5), (-np.pi,np.pi), (-100,100)]
        logger.debug(f'\n{obj_minuit.migrad()}')
        logger.debug(f'\n{obj_minuit.hesse()}')
        logger.debug(f'\n{obj_minuit.minos()}')

        chi2dof = obj_minuit.fval / self.ndof
        str_chi2 = f"𝜒²/ndof = {obj_minuit.fval:.2f} / {self.ndof} = {chi2dof}"
        logger.info(str_chi2)
        self.A = obj_minuit.values["A"]
        self.A_err = obj_minuit.errors["A"]
        self.P = self.A
        self.P_err = self.A_err
        logger.info(f'P={self.P}')
        logger.info(f'P_err={self.P_err}')
        self.ps_2phi = obj_minuit.values["ps_2phi"]
        self.ps_2phi_err = obj_minuit.errors["ps_2phi"]
        logger.info(f'phi={self.ps_2phi}, phi_err={self.ps_2phi_err}')
        logger.info(f'Q = {self.P*np.cos(self.ps_2phi)}')
        logger.info(f'U = {self.P*np.sin(self.ps_2phi)}')

    # tests
    def test_residual(self):

        m_model = np.zeros(self.npix)
        m_model[self.ipix_disc] = self.model(self.A, self.ps_2phi)

        res = self.m - m_model
        # res = res - np.load('./m_cn_b_1.npy')
        m_input = np.load('./1_6k_cn.npy')
        # m_input = np.load('./data/m_pn_b_1.npy')
        res = res - m_input


        m_min = -1.5
        m_max = 1.5
        hp.gnomview(m_input, rot=[self.pix_lon, self.pix_lat, 0], title='input cn', min=m_min, max=m_max)
        hp.gnomview(m_model, rot=[self.pix_lon, self.pix_lat, 0], title='model', min=m_min, max=m_max)
        hp.gnomview(self.m, rot=[self.pix_lon, self.pix_lat, 0], title='input pcn', min=m_min, max=m_max)
        hp.gnomview(res, rot=[self.pix_lon, self.pix_lat, 0], title='res', min=m_min, max=m_max)

        plt.show()

    # check maps
    def check_ps(self):
        hp.gnomview(self.m, rot=[self.pix_lon, self.pix_lat, 0])
        plt.show()

    def see_true_map(self, nside, beam, **kwargs):
        lon = self.lon
        lat = self.lat
        fig_size = 1000
        radiops = hp.read_map(f'/sharefs/alicpt/users/zrzhang/allFreqPSMOutput/skyinbands/AliCPT_uKCMB/{self.freq}GHz/strongradiops_map_{self.freq}GHz.fits', field=1)
        irps = hp.read_map(f'/sharefs/alicpt/users/zrzhang/allFreqPSMOutput/skyinbands/AliCPT_uKCMB/{self.freq}GHz/strongirps_map_{self.freq}GHz.fits', field=1)

        hp.gnomview(irps, rot=[lon, lat, 0], xsize=fig_size, ysize=fig_size, reso=0.3, title='irps', sub=223)
        hp.gnomview(radiops, rot=[lon, lat, 0], xsize=fig_size, ysize=fig_size, reso=0.3, title='radiops', sub=224)
        hp.gnomview(self.m, rot=[lon, lat, 0], xsize=fig_size, ysize=fig_size, reso=0.3, sub=221, title='B map')

        vec = hp.ang2vec(theta=lon, phi=lat, lonlat=True)
        ipix_disc = hp.query_disc(nside=nside, vec=vec, radius=7.5 * np.deg2rad(beam)/60)

        mask = np.ones(hp.nside2npix(nside))
        mask[ipix_disc] = 0

        hp.gnomview(mask, rot=[lon, lat, 0], xsize=fig_size, ysize=fig_size, reso=0.3, sub=222, title='mask')
        plt.show()

    def see_b_map(self, nside, beam, **kwargs):
        # lon = np.rad2deg(1.142578124999998)
        # lat = np.rad2deg(-0.09325489064144145)
        lon = self.lon
        lat = self.lat

        fig_size = 1000
        radiops = hp.read_map(f'/sharefs/alicpt/users/zrzhang/allFreqPSMOutput/skyinbands/AliCPT_uKCMB/{self.freq}GHz/strongradiops_map_{self.freq}GHz.fits', field=0)
        irps = hp.read_map(f'/sharefs/alicpt/users/zrzhang/allFreqPSMOutput/skyinbands/AliCPT_uKCMB/{self.freq}GHz/strongirps_map_{self.freq}GHz.fits', field=0)

        hp.gnomview(irps, rot=[lon, lat, 0], xsize=fig_size, ysize=fig_size, reso=0.3, title='irps', sub=223)
        hp.gnomview(radiops, rot=[lon, lat, 0], xsize=fig_size, ysize=fig_size, reso=0.3, title='radiops', sub=224)
        hp.gnomview(self.m, rot=[lon, lat, 0], xsize=fig_size, ysize=fig_size, reso=0.3, sub=221, title='B map')

        vec = hp.ang2vec(theta=lon, phi=lat, lonlat=True)
        ipix_disc = hp.query_disc(nside=nside, vec=vec, radius=12.0 * np.deg2rad(beam)/60)

        mask = np.ones(hp.nside2npix(nside))
        mask[ipix_disc] = 0

        hp.gnomview(mask, rot=[lon, lat, 0], xsize=fig_size, ysize=fig_size, reso=0.3, sub=222, title='mask')
        plt.show()

    def test_number_nearby_ps(self, threshold_extra_factor):
        num_ps, near = self.find_nearby_ps(num_ps=20, threshold_extra_factor=threshold_extra_factor)
        print(f'{num_ps=}, {near=}')
        return num_ps

    def test_lsq_params(self, *args):
        num_ps, (P2, phi2, lon2, lat2) = self.find_nearby_ps(num_ps=1, threshold_extra_factor=5)
        self.fit_lon = (self.lon, lon2)
        self.fit_lat = (self.lat, lat2)

        for P2 in np.linspace(5, 1000, 10):
            print(f'{num_ps=}, {P2=}, {phi2=}, {lon2}, {lat2}')
            args = (self.P, self.ps_2phi, P2, phi2, 0.0)
            num_ps = (len(args) - 1) // 2
            print(f'{num_ps=}')

            # Extract const
            c = 0

            # Process each point source
            models = []
            for i in range(num_ps):
                A, ps_2phi = args[i*2:i*2+2]
                lon = self.fit_lon[i]
                lat = self.fit_lat[i]
                print(f'{lon=}, {lat=}')
                lat = self.adjust_lat(lat)

                ipix_ctr = hp.ang2pix(nside=self.nside, theta=lon, phi=lat, lonlat=True)
                ctr_theta, ctr_phi = hp.pix2ang(nside=nside, ipix=ipix_ctr) # center pixel theta phi in sphere coordinate
                vec_theta = np.asarray((np.cos(ctr_theta)*np.cos(ctr_phi), np.cos(ctr_theta)*np.sin(ctr_phi), -np.sin(ctr_theta)))
                vec_phi = np.asarray((-np.sin(ctr_phi), np.cos(ctr_phi), 0))

                ctr_vec = np.array(hp.ang2vec(theta=lon, phi=lat, lonlat=True))
                vec_ctr_to_disc = self.vec_disc.T - ctr_vec # vector from center to fitting point
                r = np.linalg.norm(vec_ctr_to_disc, axis=1) # radius in polar coordinate
                # np.set_printoptions(threshold=np.inf)
                # print(f'{r=}')

                normed_vec_ctr_to_disc = vec_ctr_to_disc.T / r # normed vector from center to fitting point for calculating xi
                normed_vec_ctr_to_disc = np.nan_to_num(normed_vec_ctr_to_disc, nan=0)
                # logger.debug(f'{normed_vec_ctr_to_disc=}')

                cos_theta = normed_vec_ctr_to_disc.T @ vec_theta
                cos_phi = normed_vec_ctr_to_disc.T @ vec_phi

                xi = np.arctan2(cos_phi, cos_theta) # xi in polar coordinate
                cos_2xi = np.cos(2*xi)
                sin_2xi = np.sin(2*xi)
                # logger.debug(f'{xi=}')

                r_2 = r**2
                r_2_div_sigma = r_2 / (2 * self.sigma**2)

                model = - A * self.nside2pixarea_factor / (np.pi) * (sin_2xi * np.cos(ps_2phi) - self.cos_2xi * np.sin(ps_2phi)) * (1 / r_2) * (np.exp(-r_2_div_sigma) * (1+r_2_div_sigma) - 1)
                model = np.nan_to_num(model, nan=0)

                models.append(model)

                y_model = sum(models) + c
                y_data = self.m[self.ipix_disc]
            
                y_diff = y_data - y_model

                z = (y_diff) @ self.inv_cov @ (y_diff)
                logger.debug(f'{z=}')

if __name__=='__main__':

    df_mask = pd.read_csv('../../pp_P/mask/mask_csv/215.csv')
    df_ps = pd.read_csv('../../pp_P/mask/ps_csv/215.csv')
    lmax = 1999
    nside = 2048
    beam = 11
    freq = 215
    flux_idx = 0
    lon = df_mask.at[flux_idx, 'lon']
    print(f'{lon=}')
    lat = df_mask.at[flux_idx, 'lat']
    qflux = df_mask.at[flux_idx, 'qflux']
    uflux = df_mask.at[flux_idx, 'uflux']
    pflux = df_mask.at[flux_idx, 'pflux']

    print(f'{lon=}, {lat=}, {qflux=}, {uflux=}, {pflux=}')
    print(df_mask.head())
    # m = np.load('../../fitdata/synthesis_data/2048/PSCMBNOISE/215/1.npy').copy()
    # m = np.load('../../fitdata/2048/PS/215/ps.npy').copy()
    # m_b = hp.alm2map(hp.map2alm(m)[2], nside=nside)
    # np.save(f'./{flux_idx}.npy', m_b)
    # m_b = np.load('./1.npy')
    m_b = np.load('./1_6k_pcn.npy')


    obj = Fit_on_B(m_b, df_mask, df_ps, flux_idx, qflux, uflux, pflux, lmax, nside, beam, lon, lat, freq, r_fold=2.5, r_fold_rmv=5)
    # obj.check_ps()
    obj.params_for_fitting()
    # obj.see_true_map(nside=nside, beam=beam)
    # obj.see_b_map(nside, beam)
    # obj.calc_inv_cov(mode='n1')
    obj.calc_inv_cov(mode='cn1')
    # obj.ez_fit_b()
    # obj.fit_1_ps()
    obj.fit_2_ps()

    # obj.test_lsq_params()
    # obj.params_for_testing()
    # obj.test_residual()



