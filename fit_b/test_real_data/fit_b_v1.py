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

    def __init__(self, m, flux_idx, qflux, uflux, pflux, lmax, nside, beam, lon, lat, freq, r_fold=3, r_fold_rmv=5):
        self.m = m
        self.flux_idx = flux_idx
        self.qflux = qflux # in mJy
        self.uflux = uflux # in mJy
        self.pflux = pflux # in mJy
        self.lmax = lmax
        self.nside = nside
        self.beam = beam
        self.lon = lon # in radians
        self.lat = lat # in radians
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

        ipix_ctr = hp.ang2pix(theta=np.rad2deg(lon), phi=np.rad2deg(lat), lonlat=True, nside=nside)
        self.pix_lon, self.pix_lat = hp.pix2ang(ipix=ipix_ctr, nside=nside, lonlat=True) # lon lat in degree
        self.ctr_vec = np.array(hp.pix2vec(nside=nside, ipix=ipix_ctr))

        ctr_theta, ctr_phi = hp.pix2ang(nside=nside, ipix=ipix_ctr) # center pixel theta phi in sphere coordinate
        self.vec_theta = np.asarray((np.cos(ctr_theta)*np.cos(ctr_phi), np.cos(ctr_theta)*np.sin(ctr_phi), -np.sin(ctr_theta)))
        self.vec_phi = np.asarray((-np.sin(ctr_phi), np.cos(ctr_phi), 0))

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
            lon = np.rad2deg(self.fit_lon[i])
            lat = np.rad2deg(self.fit_lat[i])
            if np.isnan(lon): lon = self.fit_lon[i]+np.random.uniform(-0.01, 0.01)
            if np.isnan(lat): lat = self.fit_lat[i]+np.random.uniform(-0.01, 0.01)
            # print(f'{lon=},{lat=}')
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
            logger.debug(f'{normed_vec_ctr_to_disc=}')

            cos_theta = normed_vec_ctr_to_disc.T @ vec_theta
            cos_phi = normed_vec_ctr_to_disc.T @ vec_phi

            xi = np.arctan2(cos_phi, cos_theta) # xi in polar coordinate
            cos_2xi = np.cos(2*xi)
            sin_2xi = np.sin(2*xi)
            logger.debug(f'{xi=}')

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


    def test_fit_b(self):
        params = (self.P, self.ps_2phi, 0.0)
        self.fit_lon = (self.lon,)
        self.fit_lat = (self.lat,)
        logger.debug(f'{self.fit_lon=}, {self.fit_lat=}')

        obj_minuit = Minuit(self.lsq_params, name=("A", "ps_2phi", "const"), *params)
        obj_minuit.limits = [(0,1e5), (-np.pi,np.pi), (-100,100)]
        logger.debug(f'\n{obj_minuit.migrad()}')
        logger.debug(f'\n{obj_minuit.hesse()}')

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



if __name__=='__main__':

    df = pd.read_csv('../../pp_P/mask/mask_csv/215.csv')
    lmax = 1999
    nside = 2048
    beam = 11
    freq = 215
    flux_idx = 1
    lon = df.at[flux_idx, 'lon']
    print(f'{lon=}')
    lat = df.at[flux_idx, 'lat']
    qflux = df.at[flux_idx, 'qflux']
    uflux = df.at[flux_idx, 'uflux']
    pflux = df.at[flux_idx, 'pflux']

    print(f'{lon=}, {lat=}, {qflux=}, {uflux=}, {pflux=}')
    print(df.head())
    # m = np.load('../../fitdata/synthesis_data/2048/PSCMBNOISE/215/1.npy').copy()
    # m = np.load('../../fitdata/2048/PS/215/ps.npy').copy()
    # m_b = hp.alm2map(hp.map2alm(m)[2], nside=nside)
    # np.save(f'./{flux_idx}.npy', m_b)
    # m_b = np.load('./1.npy')
    m_b = np.load('./1_6k_pcn.npy')


    obj = Fit_on_B(m_b, flux_idx, qflux, uflux, pflux, lmax, nside, beam, lon, lat, freq, r_fold=2.5, r_fold_rmv=5)
    # obj.check_ps()
    obj.params_for_fitting()
    # obj.calc_inv_cov(mode='n1')
    obj.calc_inv_cov(mode='cn1')
    # obj.ez_fit_b()
    obj.test_fit_b()
    obj.params_for_testing()
    obj.test_residual()



