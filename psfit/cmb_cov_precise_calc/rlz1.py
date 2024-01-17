import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import pandas as pd
import time
import pickle
import os

class FitPointSource:
    def __init__(self, m, flux_idx, df_mask, df_ps, lon, lat, lmax, nside, radius_factor, beam):
        self.m = m # sky maps (npix,)
        self.input_lon = lon # input longitude in degrees
        self.input_lat = lat # input latitude in degrees
        ipix = hp.ang2pix(nside=nside, theta=lon, phi=lat, lonlat=True)
        lon, lat = hp.pix2ang(nside=nside, ipix=ipix, lonlat=True)
        self.lon = lon
        self.lat = lat
        print(f'{lon=}')
        print(f'{lat=}')
        self.df_mask = df_mask # pandas data frame of point sources in mask
        self.flux_idx = flux_idx # index in df_mask
        self.df_ps = df_ps # pandas data frame of all point sources
        self.lmax = lmax # maximum multipole
        self.nside = nside # resolution of healpy maps
        self.radius_factor = radius_factor # disc radius of fitting region
        self.beam = beam # arcmin

        self.sigma = np.deg2rad(beam) / 60 / (np.sqrt(8 * np.log(2)))

        ctr0_pix = hp.ang2pix(nside=self.nside, theta=self.lon, phi=self.lat, lonlat=True)
        ctr0_vec = np.array(hp.pix2vec(nside=self.nside, ipix=ctr0_pix)).astype(np.float64)

        self.ipix_fit = hp.query_disc(nside=self.nside, vec=ctr0_vec, radius=self.radius_factor * np.deg2rad(self.beam) / 60)
        self.vec_around = np.array(hp.pix2vec(nside=self.nside, ipix=self.ipix_fit.astype(int))).astype(np.float64)
        # print(f'{ipix_fit.shape=}')
        self.ndof = len(self.ipix_fit)

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

    def return_ipix_fit(self):
        return self.ipix_fit

if __name__ == '__main__':
    cmbcl = np.load('../../src/cmbsim/cmbdata/cmbcl.npy') # (n_ell, n_cl) TT, EE, BB, TE
    l = np.arange(cmbcl.shape[0])
    print(f'{cmbcl.shape}')
    
    lmax = 800
    nside = 2048
    beam = 63
    freq = 40
    
    for i in range(100):
        print(f"{i=}")
        m = hp.synfast(cmbcl.T[0], nside=nside, lmax=1000, new=True)
        print(f'{m.shape=}')
        sm = hp.smoothing(m, fwhm=np.deg2rad(beam)/60, lmax=lmax)

        df_mask = pd.read_csv('../partial_sky_ps/ps_in_mask/mask40.csv')
        df_ps = pd.read_csv('../../test/ps_sort/sort_by_iflux/40.csv')
        flux_idx = 1
        lon = np.rad2deg(df_mask.at[flux_idx, 'lon'])
        lat = np.rad2deg(df_mask.at[flux_idx, 'lat'])
        
        lmax = 350
        l = np.arange(lmax+1)

        # plt.plot(l*(l+1)*cl_cmb/(2*np.pi))
        # plt.plot(l*(l+1)*cl1/(2*np.pi), label='cl1')
        # plt.show()

        obj = FitPointSource(m=sm, flux_idx=flux_idx, df_mask=df_mask, df_ps=df_ps, lon=lon, lat=lat, lmax=lmax, nside=nside, radius_factor=1.0, beam=beam)

        obj.see_true_map(m=m, lon=lon, lat=lat, nside=nside, beam=beam)
        ipix_fit = obj.return_ipix_fit()
        my_map = sm[ipix_fit]
        print(f'{my_map.shape=}')
        np.save(f'./datarlz/{i}.npy', my_map)









