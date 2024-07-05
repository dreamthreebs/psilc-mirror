import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt

from iminuit import Minuit
from pathlib import Path

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

        nside2pixarea_factor = hp.nside2pixarea(nside=self.nside)
        self.Q = self.flux2norm_beam(qflux) / nside2pixarea_factor
        self.U = self.flux2norm_beam(uflux) / nside2pixarea_factor
        self.P = self.flux2norm_beam(pflux) / nside2pixarea_factor
        print(f'Q={self.Q}, U={self.U}, P={self.P}')

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
        print(f'{self.ipix_disc.shape=}, {self.ndof=}')

        vec_disc = np.array(hp.pix2vec(nside=self.nside, ipix=self.ipix_disc.astype(int))).astype(np.float64)
        vec_ctr_to_disc = vec_disc.T - self.ctr_vec # vector from center to fitting point

        r = np.linalg.norm(vec_ctr_to_disc, axis=1) # radius in polar coordinate
        # np.set_printoptions(threshold=np.inf)
        # print(f'{r=}')

        normed_vec_ctr_to_disc = vec_ctr_to_disc.T / r # normed vector from center to fitting point for calculating xi
        normed_vec_ctr_to_disc = np.nan_to_num(normed_vec_ctr_to_disc, nan=0)
        print(f'{normed_vec_ctr_to_disc=}')

        cos_theta = normed_vec_ctr_to_disc.T @ self.vec_theta
        cos_phi = normed_vec_ctr_to_disc.T @ self.vec_phi

        xi = np.arctan2(cos_phi, cos_theta) # xi in polar coordinate
        self.cos_2xi = np.cos(2*xi)
        self.sin_2xi = np.sin(2*xi)
        print(f'{xi=}')

        self.r_2 = r**2
        self.r_2_div_sigma = self.r_2 / (2 * self.sigma**2)

    def params_for_testing(self):
        self.ipix_disc = hp.query_disc(nside=self.nside, vec=self.ctr_vec, radius=self.r_fold_rmv * np.deg2rad(self.beam) / 60 ) # disc for fitting
        self.ndof = np.size(self.ipix_disc) # degree of freedom
        print(f'{self.ipix_disc.shape=}, {self.ndof=}')

        vec_disc = np.array(hp.pix2vec(nside=self.nside, ipix=self.ipix_disc.astype(int))).astype(np.float64)
        vec_ctr_to_disc = vec_disc.T - self.ctr_vec # vector from center to fitting point

        r = np.linalg.norm(vec_ctr_to_disc, axis=1) # radius in polar coordinate
        # np.set_printoptions(threshold=np.inf)
        # print(f'{r=}')

        normed_vec_ctr_to_disc = vec_ctr_to_disc.T / r # normed vector from center to fitting point for calculating xi
        normed_vec_ctr_to_disc = np.nan_to_num(normed_vec_ctr_to_disc, nan=0)
        print(f'{normed_vec_ctr_to_disc=}')

        cos_theta = normed_vec_ctr_to_disc.T @ self.vec_theta
        cos_phi = normed_vec_ctr_to_disc.T @ self.vec_phi

        xi = np.arctan2(cos_phi, cos_theta) # xi in polar coordinate
        self.cos_2xi = np.cos(2*xi)
        self.sin_2xi = np.sin(2*xi)
        print(f'{xi=}')

        self.r_2 = r**2
        self.r_2_div_sigma = self.r_2 / (2 * self.sigma**2)

    # Calculate inverse covariance matrix
    def calc_inv_cov(self, mode='n'):

        if mode == 'cn1':
            cmb_cov = np.load(f'./cmb_b_cov/{self.flux_idx}.npy') # load cmb cov
            # cmb_cov = np.load('./data/c_cov.npy')

            print(f'{cmb_cov.shape=}')
            # noise_cov = np.load('./data/noise_cov.npy')
            noise_cov = np.load(f'./noise_b_cov/{self.flux_idx}.npy')
            print(f'{noise_cov=}')
            cov = cmb_cov + noise_cov
            eigenval, eigenvec = np.linalg.eigh(cov)
            print(f'{eigenval=}')
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
            print(f'{eigenval=}')
            eigenval[eigenval < 0] = 1e-10
            reconstructed_cov = np.dot(eigenvec * eigenval, eigenvec.T)

            self.inv_cov = np.linalg.inv(reconstructed_cov)
            return None

        if mode == 'cn':
            cov = np.load(f'./cmb_b_cov/{self.flux_idx}.npy') # load cmb cov
            # cov = np.load('./data/c_cov.npy')
            print(f'{cov.shape=}')

            eigenval, eigenvec = np.linalg.eigh(cov)
            print(f'{eigenval=}')
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
    def model(self, A, ps_2phi):
        model = - A / (np.pi) * (self.sin_2xi * np.cos(ps_2phi) - self.cos_2xi * np.sin(ps_2phi)) * (1 / self.r_2) * (np.exp(-self.r_2_div_sigma) * (1+self.r_2_div_sigma) - 1)
        model = np.nan_to_num(model, nan=0)
        return model

    def lsq(self, A, ps_2phi, const):
        y_model = self.model(A, ps_2phi) + const
        y_data = self.m[self.ipix_disc]
        y_diff = y_data - y_model

        z = y_diff @ self.inv_cov @ y_diff
        return z

    def fit_b(self):
        params = (1e-4, 0, 0.0)
        obj_minuit = Minuit(self.lsq, name=("A", "ps_2phi", "const"), *params)
        obj_minuit.limits = [(0,1), (-np.pi,np.pi), (-100,100)]
        print(obj_minuit.migrad())
        print(obj_minuit.hesse())
        chi2dof = obj_minuit.fval / self.ndof
        str_chi2 = f"𝜒²/ndof = {obj_minuit.fval:.2f} / {self.ndof} = {chi2dof}"
        print(str_chi2)
        self.A = obj_minuit.values["A"]
        self.A_err = obj_minuit.errors["A"]
        self.P = self.A / hp.nside2pixarea(nside=self.nside)
        self.P_err = self.A_err / hp.nside2pixarea(nside=self.nside)
        print(f'P={self.P}')
        print(f'P_err={self.P_err}')
        self.ps_2phi = obj_minuit.values["ps_2phi"]
        self.ps_2phi_err = obj_minuit.errors["ps_2phi"]
        print(f'phi={self.ps_2phi}, phi_err={self.ps_2phi_err}')
        print(f'Q = {self.P*np.cos(self.ps_2phi)}')
        print(f'U = {self.P*np.sin(self.ps_2phi)}')

    # tests
    def test_residual(self):

        m_model = np.zeros(self.npix)
        m_model[self.ipix_disc] = self.model(self.A, self.ps_2phi)

        res = self.m - m_model
        # res = res - np.load('./m_cn_b_1.npy')
        m_input = np.load('./m_qu_cn.npy')
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
    m = np.load('../../fitdata/synthesis_data/2048/PSCMBNOISE/215/1.npy').copy()
    # m = np.load('../../fitdata/2048/PS/215/ps.npy').copy()
    # m_b = hp.alm2map(hp.map2alm(m)[2], nside=nside)
    # np.save(f'./{flux_idx}.npy', m_b)
    m_b = np.load('./1.npy')


    obj = Fit_on_B(m_b, flux_idx, qflux, uflux, pflux, lmax, nside, beam, lon, lat, freq, r_fold=2.5, r_fold_rmv=5)
    # obj.check_ps()
    obj.params_for_fitting()
    # obj.calc_inv_cov(mode='n1')
    obj.calc_inv_cov(mode='cn1')
    obj.fit_b()
    # obj.params_for_testing()
    # obj.test_residual()



