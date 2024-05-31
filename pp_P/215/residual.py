import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import os,sys
import pandas as pd

import ipdb
from pathlib import Path

class GetResidual:
    def __init__(self, freq, df_mask, nside, beam, radius_factor):
        self.freq = freq
        self.df_mask = df_mask
        self.nside = nside
        self.beam = beam
        self.sigma = np.deg2rad(beam) / 60 / (np.sqrt(8 * np.log(2)))
        self.radius_factor = radius_factor

    def pcn_res(self, mask, threshold=2):
        overlap_arr = np.load('./overlap_ps.npy')
        hesse_err_arr = np.array(None)
        print(f'{overlap_arr=}')
        for idx_rlz in range(100):
            print(f'{idx_rlz=}')

            m_pcn = np.load(f'../../fitdata/synthesis_data/2048/PSCMBNOISE/{self.freq}/{idx_rlz}.npy')
            m_pcn_q = m_pcn[1].copy()
            m_pcn_u = m_pcn[2].copy()

            m_cn = np.load(f'../../fitdata/synthesis_data/2048/CMBNOISE/{self.freq}/{idx_rlz}.npy')
            m_cn_q = m_cn[1].copy()
            m_cn_u = m_cn[2].copy()

            de_ps_q = m_pcn_q.copy()
            de_ps_u = m_pcn_u.copy()
            mask_list = []

            for flux_idx in range(214):
                print(f'{flux_idx=}')

                if np.in1d(flux_idx, overlap_arr):
                    print(f'this point overlap with other point')
                    continue

                if np.in1d(flux_idx, hesse_err_arr):
                    print(f'this point has hesse_err')
                    continue

                pcn_q_amp = np.load(f'./fit_res/2048/PSCMBNOISE/1.5/idx_{flux_idx}/q_amp.npy')
                pcn_u_amp = np.load(f'./fit_res/2048/PSCMBNOISE/1.5/idx_{flux_idx}/u_amp.npy')
                pcn_q_amp_err = np.load(f'./fit_res/2048/PSCMBNOISE/1.5/idx_{flux_idx}/q_amp_err.npy')
                pcn_u_amp_err = np.load(f'./fit_res/2048/PSCMBNOISE/1.5/idx_{flux_idx}/u_amp_err.npy')
                print(f'{pcn_q_amp[idx_rlz]=}, {pcn_q_amp_err[idx_rlz]=}, {pcn_u_amp[idx_rlz]=}, {pcn_u_amp_err[idx_rlz]=}')

                if (np.abs(pcn_q_amp[idx_rlz]) < threshold * pcn_q_amp_err[idx_rlz]) and (np.abs(pcn_u_amp[idx_rlz]) < threshold * pcn_u_amp_err[idx_rlz]):
                    print(f'smaller than threshold:{threshold}, pass this index')
                    continue

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
                P = (pcn_q_amp[idx_rlz] + 1j * pcn_u_amp[idx_rlz]) * np.exp(2j * self.df_mask.at[flux_idx, 'lon'])
                lugwid_P = P * profile
                QU = lugwid_P * np.exp(-2j * phi_around)
                Q = QU.real
                U = QU.imag

                # hp.gnomview(m_cn_q, rot=[pcn_fit_lon, pcn_fit_lat, 0], xsize=30, ysize=30, title='cmb noise q')
                # hp.gnomview(m_cn_u, rot=[pcn_fit_lon, pcn_fit_lat, 0], xsize=30, ysize=30, title='cmb noise u')
                # plt.show()

                # hp.gnomview(de_ps_q, rot=[pcn_fit_lon, pcn_fit_lat, 0], xsize=30, ysize=30, title='before removal q')
                # hp.gnomview(de_ps_u, rot=[pcn_fit_lon, pcn_fit_lat, 0], xsize=30, ysize=30, title='before removal u')
                # plt.show()

                de_ps_q[ipix_fit] = de_ps_q[ipix_fit].copy() - Q
                de_ps_u[ipix_fit] = de_ps_u[ipix_fit].copy() - U

                # hp.gnomview(de_ps_q, rot=[pcn_fit_lon, pcn_fit_lat, 0], xsize=30, ysize=30, title='after removal q')
                # hp.gnomview(de_ps_u, rot=[pcn_fit_lon, pcn_fit_lat, 0], xsize=30, ysize=30, title='after removal u')
                # plt.show()

            res_q = np.copy(de_ps_q)
            res_u = np.copy(de_ps_u)

            path_for_res_map = Path(f'./fit_res/2048/pcn_after_removal/{threshold}sigma')
            path_for_res_map.mkdir(parents=True, exist_ok=True)
            np.save(path_for_res_map / Path(f'map_q_{idx_rlz}.npy'), res_q)
            np.save(path_for_res_map / Path(f'map_u_{idx_rlz}.npy'), res_u)
            np.save(path_for_res_map / Path(f'mask_{idx_rlz}.npy'), np.array(mask_list))

def main():
    freq = 215
    nside = 2048
    beam = 11
    df_mask = pd.read_csv(f'../mask/mask_csv/{freq}.csv')
    radius_factor = 1.5
    mask = np.load('../../src/mask/north/BINMASKG2048.npy')

    obj = GetResidual(freq=freq, df_mask=df_mask, nside=nside, beam=beam, radius_factor=radius_factor)
    obj.pcn_res(mask, threshold=3)


main()

