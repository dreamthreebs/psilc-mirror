import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import os,sys
import pandas as pd

from pathlib import Path

class GetResidual:
    def __init__(self, flux_idx, m_has_ps, m_no_ps, lon, lat, df_mask, nside, beam, radius_factor):
        self.flux_idx = flux_idx
        self.m_has_ps = m_has_ps
        self.m_no_ps = m_no_ps

        self.lon = lon
        self.lat = lat
        self.nside = nside
        self.beam = beam
        self.sigma = np.deg2rad(beam) / 60 / (np.sqrt(8 * np.log(2)))
        self.df_mask = df_mask
        self.radius_factor = radius_factor
        iflux = df_mask.at[flux_idx, 'iflux']
        self.true_beam = self.flux2norm_beam(iflux)

        ctr0_pix = hp.ang2pix(nside=self.nside, theta=self.lon, phi=self.lat, lonlat=True)
        ctr0_vec = np.array(hp.pix2vec(nside=self.nside, ipix=ctr0_pix)).astype(np.float64)

        self.ipix_fit = hp.query_disc(nside=self.nside, vec=ctr0_vec, radius=self.radius_factor * np.deg2rad(self.beam) / 60)
        self.vec_around = np.array(hp.pix2vec(nside=self.nside, ipix=self.ipix_fit.astype(int))).astype(np.float64)
        # print(f'{ipix_fit.shape=}')
        self.ndof = len(self.ipix_fit)

    def beam_model(self, norm_beam, theta):
        return norm_beam / (2 * np.pi * self.sigma**2) * np.exp(- (theta)**2 / (2 * self.sigma**2))

    def flux2norm_beam(self, flux):
        # from mJy to muK_CMB to norm_beam
        coeffmJy2norm = 2.1198465131100624e-05
        return coeffmJy2norm * flux

    def pscmbnoise(self, mask):
        res_list = []
        n_rlz = 100
        for i in range(n_rlz):
            res = np.load(f'./fit_res/2048/ps_cmb_noise_residual/{i}.npy')[self.ipix_fit].copy()
            print(f'{res.shape=}')
            res_list.append(res)

        res_arr = np.asarray(res_list)
        mean = np.mean(res_arr, axis=0)
        print(f'{np.max(mean)=}')
        rms = np.sqrt(np.sum(res_arr**2, axis=0) / n_rlz)
        print(f'{rms=}')

        rms_map = np.zeros_like(self.m_has_ps)
        rms_map[self.ipix_fit] = rms

        hp.gnomview(rms_map, rot=[self.lon, self.lat, 0], xsize=100, ysize=100, title='rms residual map')
        plt.show()

        # plt.plot(np.arange(len(self.ipix_fit)), var)
        # plt.show()

    def pscmbnoise_avg(self, mask):

        res_list = []

        n_rlz = 100
        for i in range(n_rlz):
            res = np.load(f'./fit_res/2048/ps_cmb_fg_noise_residual/{i}.npy')[self.ipix_fit].copy()
            print(f'{res.shape=}')
            res_list.append(res)

        res_arr = np.asarray(res_list)
        # np.save('./fit_res/2048/ps_cmb_fg_noise_residual/all.npy', res_arr)

        # res_arr = np.load('./fit_res/2048/ps_cmb_noise_residual/all.npy')

        rms = np.mean(np.sqrt(np.sum(res_arr**2, axis=1) / len(self.ipix_fit)))
        print(f'{rms=}')

        # mean_rms = np.mean(rms)
        # print(f'{mean_rms=}')
        return rms

    def pscmbfgnoise_avg(self, mask):

        res_list = []

        n_rlz = 100
        for i in range(n_rlz):
            res = np.load(f'./fit_res/2048/ps_cmb_fg_noise_residual/{i}.npy')[self.ipix_fit].copy()
            print(f'{res.shape=}')
            res_list.append(res)

        res_arr = np.asarray(res_list)
        # np.save('./fit_res/2048/ps_cmb_fg_noise_residual/all.npy', res_arr)

        # res_arr = np.load('./fit_res/2048/ps_cmb_fg_noise_residual/all.npy')

        rms = np.mean(np.sqrt(np.sum(res_arr**2, axis=1) / len(self.ipix_fit)))
        print(f'{rms=}')

        # mean_rms = np.mean(rms)
        # print(f'{mean_rms=}')
        return rms






def main():
    # m = np.load('../../fitdata/synthesis_data/2048/PSNOISE/40/1.npy')[0]
    m_has_ps = np.load('../../fitdata/synthesis_data/2048/PSNOISE/40/1.npy')[0]
    # m_no_ps = np.load('../../fitdata/2048/NOISE/40/1.npy')[0]
    m_no_ps = None
    mask = np.load('../../src/mask/north/BINMASKG2048.npy')
    nstd = np.load('../../FGSim/NSTDNORTH/2048/40.npy')[0]
    df_mask = pd.read_csv('../partial_sky_ps/ps_in_mask/2048/40mask.csv')
    df_ps = pd.read_csv('../partial_sky_ps/ps_in_mask/2048/40ps.csv')
    lmax = 350
    nside = 2048
    beam = 63

    flux_idx = 1
    lon = np.rad2deg(df_mask.at[flux_idx, 'lon'])
    lat = np.rad2deg(df_mask.at[flux_idx, 'lat'])

    obj = GetResidual(flux_idx=flux_idx, m_has_ps=m_has_ps, m_no_ps=m_no_ps, lon=lon, lat=lat, df_mask=df_mask, nside=nside, beam=beam, radius_factor=1.5)

    # obj.see_true_map(m=m, lon=lon, lat=lat, nside=nside, beam=beam)
    # obj.psnoise(mask=mask)
    # obj.pscmbnoise(mask=mask)
    obj.pscmbnoise_avg(mask=mask)


def main_all():
    # m = np.load('../../fitdata/synthesis_data/2048/PSNOISE/40/1.npy')[0]
    m_has_ps = np.load('../../fitdata/synthesis_data/2048/PSNOISE/40/1.npy')[0]
    # m_no_ps = np.load('../../fitdata/2048/NOISE/40/1.npy')[0]
    m_no_ps = None
    mask = np.load('../../src/mask/north/BINMASKG2048.npy')
    nstd = np.load('../../FGSim/NSTDNORTH/2048/40.npy')[0]
    df_mask = pd.read_csv('../partial_sky_ps/ps_in_mask/2048/40mask.csv')
    df_ps = pd.read_csv('../partial_sky_ps/ps_in_mask/2048/40ps.csv')
    lmax = 350
    nside = 2048
    beam = 63

    rms_list = []
    for flux_idx in range(136):
        lon = np.rad2deg(df_mask.at[flux_idx, 'lon'])
        lat = np.rad2deg(df_mask.at[flux_idx, 'lat'])

        obj = GetResidual(flux_idx=flux_idx, m_has_ps=m_has_ps, m_no_ps=m_no_ps, lon=lon, lat=lat, df_mask=df_mask, nside=nside, beam=beam, radius_factor=1.5)

        # obj.see_true_map(m=m, lon=lon, lat=lat, nside=nside, beam=beam)
        # obj.psnoise(mask=mask)
        # obj.pscmbnoise(mask=mask)
        rms = obj.pscmbnoise_avg(mask=mask)
        print(f'{rms=}')
        rms_list.append(rms)

    np.save('./fit_res/2048/ps_cmb_noise_residual/rms.npy', rms_list)


# main()
main_all()
