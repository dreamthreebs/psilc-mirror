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
    def __init__(self, freq, df_mask, nside, beam, radius_factor):
        self.freq = freq
        self.df_mask = df_mask
        self.nside = nside
        self.beam = beam
        self.sigma = np.deg2rad(beam) / 60 / (np.sqrt(8 * np.log(2)))
        self.radius_factor = radius_factor
        self.nside2pixarea_factor = hp.nside2pixarea(nside=self.nside)

    def gen_map(self, rlz_idx):
        nside = self.nside
        npix = hp.nside2npix(nside=self.nside)
        beam = self.beam

        ps = np.load('./data/ps/ps.npy')

        nstd = np.load('../../FGSim/NSTDNORTH/2048/215.npy')
        np.random.seed(seed=noise_seeds[rlz_idx])
        # noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))
        noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))
        print(f"{np.std(noise[1])=}")

        # cmb_iqu = np.load(f'../../fitdata/2048/CMB/215/{rlz_idx}.npy')
        # cls = np.load('../../src/cmbsim/cmbdata/cmbcl.npy')
        cls = np.load('../../src/cmbsim/cmbdata/cmbcl_8k.npy')
        np.random.seed(seed=cmb_seeds[rlz_idx])
        # cmb_iqu = hp.synfast(cls.T, nside=nside, fwhm=np.deg2rad(beam)/60, new=True, lmax=1999)
        cmb_iqu = hp.synfast(cls.T, nside=nside, fwhm=np.deg2rad(beam)/60, new=True, lmax=3*nside-1)

        pcn = noise + ps + cmb_iqu
        cn = noise + cmb_iqu

        # m = np.load('./1_8k.npy')
        # np.save('./1_6k_pcn.npy', m)
        return pcn, cn



    def pcn_res(self, mask, threshold=2):
        overlap_arr = np.array(None)
        hesse_err_arr = np.array(None)
        print(f'{overlap_arr=}')
        rlz_idx=0
        print(f'{rlz_idx=}')

        m_pcn, m_cn = self.gen_map(rlz_idx=rlz_idx)
        m_pcn_q = m_pcn[1].copy()
        m_pcn_u = m_pcn[2].copy()

        m_cn_q = m_cn[1].copy()
        m_cn_u = m_cn[2].copy()

        de_ps_q = m_pcn_q.copy()
        de_ps_u = m_pcn_u.copy()
        mask_list = []

        for flux_idx in range(20):
            print(f'{flux_idx=}')

            if np.in1d(flux_idx, overlap_arr):
                print(f'this point overlap with other point')
                continue

            if np.in1d(flux_idx, hesse_err_arr):
                print(f'this point has hesse_err')
                continue

            pcn_p_amp = np.load(f'./fit_res/pcn_params/fit_b/idx_{flux_idx}/fit_P_{rlz_idx}.npy')
            pcn_phi = np.load(f'./fit_res/pcn_params/fit_b/idx_{flux_idx}/fit_phi_{rlz_idx}.npy')

            pcn_q_amp = pcn_p_amp * np.cos(pcn_phi)
            pcn_u_amp = pcn_p_amp * np.sin(pcn_phi)

            pcn_p_amp_true = np.load(f'./fit_res/pcn_params/fit_qu/idx_{flux_idx}/P_{rlz_idx}.npy')
            pcn_phi_true = np.load(f'./fit_res/pcn_params/fit_qu/idx_{flux_idx}/phi_{rlz_idx}.npy')

            pcn_q_amp_true = pcn_p_amp_true * np.cos(pcn_phi_true)
            pcn_u_amp_true = pcn_p_amp_true * np.sin(pcn_phi_true)

            print(f'{pcn_p_amp=}, {pcn_p_amp_true=}, {pcn_phi=}, {pcn_phi_true=}')
            print(f'{pcn_q_amp=}, {pcn_q_amp_true=}, {pcn_u_amp=}, {pcn_u_amp_true=}')

            # if (np.abs(pcn_q_amp[rlz_idx]) < threshold * pcn_q_amp_err[rlz_idx]) and (np.abs(pcn_u_amp[rlz_idx]) < threshold * pcn_u_amp_err[rlz_idx]):
            #     print(f'smaller than threshold:{threshold}, pass this index')
            #     continue

            mask_list.append(flux_idx)

            pcn_fit_lon = np.rad2deg(self.df_mask.at[flux_idx, 'lon'])
            pcn_fit_lat = np.rad2deg(self.df_mask.at[flux_idx, 'lat'])

            ctr0_pix = hp.ang2pix(nside=self.nside, theta=pcn_fit_lon, phi=pcn_fit_lat, lonlat=True)
            ctr0_vec = np.array(hp.pix2vec(nside=self.nside, ipix=ctr0_pix)).astype(np.float64)

            ipix_fit = hp.query_disc(nside=self.nside, vec=ctr0_vec, radius=self.radius_factor * np.deg2rad(self.beam) / 60)
            vec_around = np.array(hp.pix2vec(nside=self.nside, ipix=ipix_fit.astype(int))).astype(np.float64)
            _, phi_around = hp.pix2ang(nside=self.nside, ipix=ipix_fit)
            pcn_vec = np.asarray(hp.ang2vec(theta=pcn_fit_lon, phi=pcn_fit_lat, lonlat=True))
            cos_theta = pcn_vec @ vec_around
            cos_theta = np.clip(cos_theta, -1, 1)
            theta = np.arccos(cos_theta) # (n_rlz, n_pix_for_fit)

            profile = 1 / (2 * np.pi * self.sigma**2) * np.exp(- (theta)**2 / (2 * self.sigma**2))
            P = (pcn_q_amp + 1j * pcn_u_amp) * self.nside2pixarea_factor * np.exp(2j * self.df_mask.at[flux_idx, 'lon'])
            lugwid_P = P * profile
            QU = lugwid_P * np.exp(-2j * phi_around)
            Q = QU.real
            U = QU.imag

            # hp.gnomview(m_cn_q, rot=[pcn_fit_lon, pcn_fit_lat, 0], xsize=100, ysize=100, title='cmb noise q')
            # hp.gnomview(m_cn_u, rot=[pcn_fit_lon, pcn_fit_lat, 0], xsize=100, ysize=100, title='cmb noise u')
            # # plt.show()

            # hp.gnomview(de_ps_q, rot=[pcn_fit_lon, pcn_fit_lat, 0], xsize=100, ysize=100, title='before removal q')
            # hp.gnomview(de_ps_u, rot=[pcn_fit_lon, pcn_fit_lat, 0], xsize=100, ysize=100, title='before removal u')
            # # plt.show()

            de_ps_q[ipix_fit] = de_ps_q[ipix_fit].copy() - Q
            de_ps_u[ipix_fit] = de_ps_u[ipix_fit].copy() - U

            # hp.gnomview(de_ps_q, rot=[pcn_fit_lon, pcn_fit_lat, 0], xsize=100, ysize=100, title='after removal q')
            # hp.gnomview(de_ps_u, rot=[pcn_fit_lon, pcn_fit_lat, 0], xsize=100, ysize=100, title='after removal u')
            # hp.gnomview(de_ps_q-m_cn_q, rot=[pcn_fit_lon, pcn_fit_lat, 0], xsize=100, ysize=100, title='after removal q res')
            # hp.gnomview(de_ps_u-m_cn_u, rot=[pcn_fit_lon, pcn_fit_lat, 0], xsize=100, ysize=100, title='after removal u res')
            # plt.show()

        res_q = np.copy(de_ps_q)
        res_u = np.copy(de_ps_u)

        path_for_res_map = Path(f'./fit_res/pcn_fit_b_qu/{threshold}sigma')
        path_for_res_map.mkdir(parents=True, exist_ok=True)
        np.save(path_for_res_map / Path(f'map_q_{rlz_idx}.npy'), res_q)
        np.save(path_for_res_map / Path(f'map_u_{rlz_idx}.npy'), res_u)
        np.save(path_for_res_map / Path(f'mask_{rlz_idx}.npy'), np.array(mask_list))

def main():
    freq = 215
    nside = 2048
    beam = 11
    df_mask = pd.read_csv(f'./mask/215.csv')
    radius_factor = 1.5
    mask = np.load('../../src/mask/north/BINMASKG2048.npy')

    obj = GetResidual(freq=freq, df_mask=df_mask, nside=nside, beam=beam, radius_factor=radius_factor)
    obj.pcn_res(mask, threshold=3)


main()


