import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import os,sys
import pandas as pd

import ipdb
from pathlib import Path

noise_seeds = np.load('./seeds_noise_2k.npy')
cmb_seeds = np.load('./seeds_cmb_2k.npy')

class GetResidual:
    def __init__(self, freq, df_mask, nside, beam, r_fold_rmv):
        self.freq = freq
        self.df_mask = df_mask
        self.nside = nside
        self.beam = beam
        self.sigma = np.deg2rad(beam) / 60 / (np.sqrt(8 * np.log(2)))
        self.r_fold_rmv = r_fold_rmv
        self.nside2pixarea_factor = hp.nside2pixarea(nside=nside)

    def params_for_rmv(self):
        ipix_ctr = hp.ang2pix(theta=self.lon, phi=self.lat, lonlat=True, nside=self.nside)
        self.pix_lon, self.pix_lat = hp.pix2ang(ipix=ipix_ctr, nside=self.nside, lonlat=True) # lon lat in degree
        self.ctr_vec = np.array(hp.pix2vec(nside=self.nside, ipix=ipix_ctr))

        ctr_theta, ctr_phi = hp.pix2ang(nside=self.nside, ipix=ipix_ctr) # center pixel theta phi in sphere coordinate

        self.vec_theta = np.asarray((np.cos(ctr_theta)*np.cos(ctr_phi), np.cos(ctr_theta)*np.sin(ctr_phi), -np.sin(ctr_theta)))
        self.vec_phi = np.asarray((-np.sin(ctr_phi), np.cos(ctr_phi), 0))

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

    def gen_b_map(self, rlz_idx):
        nside = self.nside
        npix = hp.nside2npix(nside=self.nside)
        beam = self.beam

        ps = np.load('./data/ps/ps_b.npy')

        nstd = np.load('../../FGSim/NSTDNORTH/2048/215.npy')
        np.random.seed(seed=noise_seeds[rlz_idx])
        # noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))
        noise = nstd[1] * np.random.normal(loc=0, scale=1, size=(npix,))
        print(f"{np.std(noise[1])=}")

        # cmb_iqu = np.load(f'../../fitdata/2048/CMB/215/{rlz_idx}.npy')
        # cls = np.load('../../src/cmbsim/cmbdata/cmbcl.npy')
        cls = np.load('../../src/cmbsim/cmbdata/cmbcl_8k.npy')
        np.random.seed(seed=cmb_seeds[rlz_idx])
        # cmb_iqu = hp.synfast(cls.T, nside=nside, fwhm=np.deg2rad(beam)/60, new=True, lmax=1999)
        cmb_iqu = hp.synfast(cls.T, nside=nside, fwhm=np.deg2rad(beam)/60, new=True, lmax=3*nside-1)
        cmb_b = hp.alm2map(hp.map2alm(cmb_iqu)[2], nside=nside)

        pcn_b = noise + ps + cmb_b
        cn_b = noise + cmb_b

        # m = np.load('./1_8k.npy')
        # np.save('./1_6k_pcn.npy', m)
        return pcn_b, cn_b

    def model(self, A, ps_2phi):
        model = - A * self.nside2pixarea_factor / (np.pi) * (self.sin_2xi * np.cos(ps_2phi) - self.cos_2xi * np.sin(ps_2phi)) * (1 / self.r_2) * (np.exp(-self.r_2_div_sigma) * (1+self.r_2_div_sigma) - 1)
        model = np.nan_to_num(model, nan=0)
        return model

    def pcn_res(self, threshold=2):
        rlz_idx=0
        overlap_arr = np.array(None)
        hesse_err_arr = np.array(None)
        print(f'{overlap_arr=}')
        print(f'{rlz_idx=}')

        m_pcn_b, m_cn_b = self.gen_b_map(rlz_idx=rlz_idx)
        de_ps_b = m_pcn_b.copy()
        mask_list = []

        for flux_idx in range(20):
            print(f'{flux_idx=}')
            self.lon = np.rad2deg(self.df_mask.at[flux_idx, 'lon'])
            self.lat = np.rad2deg(self.df_mask.at[flux_idx, 'lat'])
            self.params_for_rmv()

            if np.in1d(flux_idx, overlap_arr):
                print(f'this point overlap with other point')
                continue

            if np.in1d(flux_idx, hesse_err_arr):
                print(f'this point has hesse_err')
                continue

            pcn_p_amp = np.load(f'./fit_res/pcn_params/fit_1/idx_{flux_idx}/fit_P_{rlz_idx}.npy')
            pcn_phi = np.load(f'./fit_res/pcn_params/fit_1/idx_{flux_idx}/fit_phi_{rlz_idx}.npy')

            pcn_p_amp_true = np.load(f'./fit_res/pcn_params/fit_1/idx_{flux_idx}/P_{rlz_idx}.npy')
            pcn_phi_true = np.load(f'./fit_res/pcn_params/fit_1/idx_{flux_idx}/phi_{rlz_idx}.npy')

            # pcn_p_amp_err = np.load(f'./fit_res/pcn_params/idx_{flux_idx}/fit_P_err_{rlz_idx}.npy')
            # pcn_phi_err = np.load(f'./fit_res/pcn_params/idx_{flux_idx}/fit_phi_err_{rlz_idx}.npy')
            # print(f'{pcn_p_amp=}, {pcn_p_amp_true=}, {pcn_p_amp_err=}, {pcn_phi=}, {pcn_phi_true=}, {pcn_phi_err=}')
            print(f'{pcn_p_amp=}, {pcn_p_amp_true=}, {pcn_phi=}, {pcn_phi_true=}')

            # if (pcn_p_amp < threshold * pcn_p_amp_err) or (pcn_phi_err > 0.15):
            #     print(f'smaller than threshold:{threshold}, pass this index')
            #     continue

            mask_list.append(flux_idx)

            m_local = self.model(A=pcn_p_amp, ps_2phi=pcn_phi)

            de_ps_b[self.ipix_disc] = de_ps_b[self.ipix_disc].copy() - m_local

            # hp.gnomview(de_ps_b, rot=[self.lon, self.lat, 0], xsize=100, ysize=100, title='after removal b')
            # hp.gnomview(de_ps_b - m_cn_b, rot=[self.lon, self.lat, 0], xsize=100, ysize=100, title='res_b')
            # hp.gnomview(m_cn_b, rot=[self.lon, self.lat, 0], xsize=100, ysize=100, title='cn b')
            # hp.gnomview(m_pcn_b, rot=[self.lon, self.lat, 0], xsize=100, ysize=100, title='pcn b')
            # plt.show()

        res_b = np.copy(de_ps_b)

        path_for_res_map = Path(f'./fit_res/pcn_fit_b/{threshold}sigma')
        path_for_res_map.mkdir(parents=True, exist_ok=True)
        np.save(path_for_res_map / Path(f'map_b_{rlz_idx}.npy'), res_b)
        np.save(path_for_res_map / Path(f'mask_{rlz_idx}.npy'), np.array(mask_list))

def main():
    freq = 215
    df_mask = pd.read_csv('../../pp_P/mask/mask_csv/215.csv')
    nside = 2048
    beam = 11
    r_fold_rmv = 5

    obj = GetResidual(freq, df_mask, nside, beam, r_fold_rmv)
    obj.pcn_res(threshold=3)

main()



