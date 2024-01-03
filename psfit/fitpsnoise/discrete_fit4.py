import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pandas as pd
import time
import pickle

from iminuit import Minuit
from iminuit.cost import LeastSquares
from numpy.polynomial.legendre import Legendre
from scipy.interpolate import CubicSpline

class FitPointSource:
    '''
    stop finding minimum value if chi2/dof satisfy command
    '''
    def __init__(self, m, nstd, flux_idx, df_mask, df_ps, cl_cmb, lon, lat, iflux, lmax, nside, radius_factor, beam, sigma_threshold=5):
        self.m = m # sky maps (npix,)
        self.input_lon = lon # input longitude in degrees
        self.input_lat = lat # input latitude in degrees
        self.ini_ipix = hp.ang2pix(nside=nside, theta=lon, phi=lat, lonlat=True)
        print(f'{self.ini_ipix=}')
        lon, lat = hp.pix2ang(nside=nside, ipix=self.ini_ipix, lonlat=True)
        self.lon = lon
        self.lat = lat
        print(f'{lon=}')
        print(f'{lat=}')
        self.iflux = iflux # temperature flux in muK_CMB
        self.df_mask = df_mask # pandas data frame of point sources in mask
        self.flux_idx = flux_idx # index in df_mask
        self.df_ps = df_ps # pandas data frame of all point sources
        self.nstd = nstd # noise standard deviation
        self.cl_cmb = cl_cmb # power spectrum of CMB
        self.lmax = lmax # maximum multipole
        self.nside = nside # resolution of healpy maps
        self.radius_factor = radius_factor # disc radius of fitting region
        self.sigma_threshold = sigma_threshold # judge if a signal is a point source
        self.beam = beam # arcmin

        self.ini_norm_beam = self.flux2norm_beam(self.iflux)

        self.n_pts = 0 # count number for counting
        self.n_call = 0.0 # count number of function calls

        self.sigma = np.deg2rad(beam) / 60 / (np.sqrt(8 * np.log(2)))

        self.do_3_params = False
        self.do_4_params = False
        self.do_5_params = False

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

    def find_nearby_ps(self, threshold_factor=2.2):
        dir_0 = (self.lon, self.lat)
        arr_1 = self.df_ps.loc[:, 'Unnamed: 0']
        print(f'{arr_1.shape=}')
        bool_arr = self.df_ps.loc[:, 'Unnamed: 0'] != self.df_mask.at[self.flux_idx, 'Unnamed: 0']
        print(f'{bool_arr.shape=}')
        lon_other = np.rad2deg(self.df_ps.loc[bool_arr, 'lon'])
        lat_other = np.rad2deg(self.df_ps.loc[bool_arr, 'lat'])
        dir_other = (lon_other, lat_other)
        ang = np.rad2deg(hp.rotator.angdist(dir1=dir_0, dir2=dir_other, lonlat=True))
        print(f'{ang.shape=}')
        threshold = threshold_factor * self.beam / 60
        print(f'{ang=}')
    
        index_near = np.nonzero(np.where((ang < threshold), ang, 0))
        ang_near = ang[index_near]
        print(f'{index_near=}')
        print(f'{ang_near=}')
        if index_near[0].size == 0:
            return 0, None

        num_ps = index_near[0].size
        print(f'{num_ps=}')

        lon_list = []
        lat_list = []
        iflux_list = []
        for i in range(min(num_ps, len(index_near[0]))):
            index = index_near[0][i]
            if index < self.df_mask.at[self.flux_idx, 'Unnamed: 0']:
                lon = np.rad2deg(self.df_ps.at[index, 'lon'])
                lat = np.rad2deg(self.df_ps.at[index, 'lat'])
                iflux = self.flux2norm_beam(self.df_ps.at[index, 'iflux'])
            else:
                lon = np.rad2deg(self.df_ps.at[index + 1, 'lon'])
                lat = np.rad2deg(self.df_ps.at[index + 1, 'lat'])
                iflux = self.flux2norm_beam(self.df_ps.at[index + 1, 'iflux'])
            lon, lat = self.input_lonlat2pix_lonlat(lon, lat)
            lon_list.append(lon)
            lat_list.append(lat)
            iflux_list.append(iflux)

        print(f'{iflux_list=}')
    
        ##Optional visualization code commented out
        #hp.gnomview(self.m, rot=[self.lon,self.lat,0])
        #for lon, lat in zip(lon_list, lat_list):
        #    hp.projscatter(lon, lat, lonlat=True)
        #plt.show()
    
        # return tuple(iflux_list + lon_list + lat_list)
        return num_ps, tuple(sum(zip(iflux_list, lon_list, lat_list), ()))

    @staticmethod
    def single_ps_model(norm_beam, sigma, theta, const):
        return norm_beam / (2 * np.pi * sigma**2) * np.exp(- (theta)**2 / (2 * sigma**2)) + const
    @staticmethod
    def alternate_loop_with_index(array):
        n = len(array)
        mid = n // 2  # find middle index
        yield mid, array[mid]

        left, right = mid - 1, mid + 1

        while left >= 0 or right < n:
            if right < n:
                yield right, array[right]
                right += 1

            if left >= 0:
                yield left, array[left]
                left -= 1

    def fit_ps_ns(self, mode:str='pipeline', ps_pos_bias=0.05):
        def lsq_2_params(norm_beam, const):
            ctr_vec = np.array(hp.pix2vec(nside=self.nside, ipix=idx))

            theta = hp.rotator.angdist(dir1=ctr_vec, dir2=vec_around)
            y_model = FitPointSource.single_ps_model(norm_beam=norm_beam,sigma=self.sigma, theta=theta, const=const)
            y_data = self.m[ipix_fit]
            y_err = self.nstd[ipix_fit]

            z = (y_data - y_model) / y_err
            return np.sum(z**2)
        def lsq_ps(*args):
            num_ps = (len(args) - 1)
            const = args[-1]
            thetas = []
            for i in range(num_ps):
                norm_beam = args[i]
                ctr_vec = np.array(hp.pix2vec(nside=self.nside, ipix=self.ctr_ipix_disc[i]))

                theta = hp.rotator.angdist(dir1=ctr_vec, dir2=vec_around)
                thetas.append(norm_beam / (2 * np.pi * self.sigma**2) * np.exp(- (theta)**2 / (2 * self.sigma**2)))

            def model():
                return sum(thetas) + const

            y_model = model()
            y_data = self.m[ipix_fit]
            y_err = self.nstd[ipix_fit]

            z = (y_data - y_model) / y_err
            return np.sum(z**2)

        num_ps, near = self.find_nearby_ps()
        print(f'{near=}')

        ctr0_vec = hp.ang2vec(theta=self.lon, phi=self.lat, lonlat=True)
        ctr1_ipix_disc = hp.query_disc(nside=self.nside, vec=ctr0_vec, radius=ps_pos_bias * np.deg2rad(self.beam) / 60)
        ipix_fit = hp.query_disc(nside=self.nside, vec=ctr0_vec, radius=self.radius_factor * np.deg2rad(self.beam) / 60)
        vec_around = np.array(hp.pix2vec(nside=self.nside, ipix=ipix_fit))
        ndof = len(ipix_fit)

        print(f'{ctr1_ipix_disc.shape=}')
        print(f'{ipix_fit.shape=}')
        fit_result_list = []
        for idx in ctr1_ipix_disc:

            obj_minuit = Minuit(lsq_2_params, norm_beam=self.ini_norm_beam, const=0.0)
            obj_minuit.limits = [(0,1),  (-100,100)]
            obj_minuit.migrad()
            obj_minuit.hesse()
            print(f'{obj_minuit.values}')
            print(f'{obj_minuit.errors}')
            chi2dof = obj_minuit.fval / ndof
            str_chi2 = f"𝜒²/ndof = {obj_minuit.fval:.2f} / {ndof} = {chi2dof}"
            print(str_chi2)
            fit_result_list.append(chi2dof)

        fit_res_arr = np.array(fit_result_list)
        print(f'{fit_result_list=}')
        min_chi2dof = np.min(fit_res_arr)
        min_chi2dof_idx = np.argmin(fit_res_arr)
        print(f'minimum idx is {min_chi2dof_idx}, chi2dof is {min_chi2dof}')

        idx = ctr1_ipix_disc[min_chi2dof_idx]
        print(f'minimum chi2dof ipix = {idx}')
        obj_minuit = Minuit(lsq_2_params, norm_beam=self.ini_norm_beam, const=0.0)
        obj_minuit.limits = [(0,1),  (-100,100)]
        print(obj_minuit.migrad())
        print(obj_minuit.hesse())
        chi2dof = obj_minuit.fval / ndof
        str_chi2 = f"𝜒²/ndof = {obj_minuit.fval:.2f} / {ndof} = {chi2dof}"
        print(str_chi2)
        is_ps = obj_minuit.values['norm_beam'] > self.sigma_threshold * obj_minuit.errors['norm_beam']
        if obj_minuit.fmin.hesse_failed:
            raise ValueError('hesse failed !!!')

        if chi2dof < 1.065:
            if is_ps:
                print(f'2 parameter fitting is enough, hesse ok!')
                self.fit_norm_beam = obj_minuit.values['norm_beam']
                fit_lon, fit_lat = np.deg2rad(hp.pix2ang(nside=self.nside, ipix=idx, lonlat=True))
                print(f'{self.ini_norm_beam=}')
                print(f'{self.fit_norm_beam=}')
                fit_error = np.abs((self.fit_norm_beam - self.ini_norm_beam) / self.ini_norm_beam)
                print(f'{fit_error=}')
                return self.fit_norm_beam, fit_lon, fit_lat
            else:
                print(f'not a point source under {self.sigma_threshold} sigma threshold')
                return 0.0, np.deg2rad(self.lon), np.deg2rad(self.lat)
        else:
            ctr2_iflux, ctr2_lon, ctr2_lat = near[0], near[1], near[2]
            ctr2_ipix = hp.ang2pix(nside=self.nside, theta=ctr2_lon, phi=ctr2_lat, lonlat=True)
            self.ctr_ipix_disc = (self.ini_ipix, ctr2_ipix)
            ini_params = (self.ini_norm_beam, ctr2_iflux, 0.0)

            obj_minuit = Minuit(lsq_ps, name=("norm_beam1", "norm_beam2", "const"), *ini_params)
            obj_minuit.limits = [(0,1.0), (0,1.0), (-100,100)]
            obj_minuit.migrad()
            obj_minuit.hesse()

            chi2dof = obj_minuit.fval / ndof
            str_chi2 = f"𝜒²/ndof = {obj_minuit.fval:.2f} / {ndof} = {chi2dof}"
            print(str_chi2)
            if chi2dof > 1.065:
                ctr2_iflux, ctr2_lon, ctr2_lat, ctr3_iflux, ctr3_lon, ctr3_lat= near[0], near[1], near[2], near[3], near[4], near[5]

                ctr2_ipix = hp.ang2pix(nside=self.nside, theta=ctr2_lon, phi=ctr2_lat, lonlat=True)
                ctr3_ipix = hp.ang2pix(nside=self.nside, theta=ctr3_lon, phi=ctr3_lat, lonlat=True)

                self.ctr_ipix_disc = (self.ini_ipix, ctr2_ipix, ctr3_ipix)
                ini_params = (self.ini_norm_beam, ctr2_iflux, ctr3_iflux, 0.0)
                obj_minuit = Minuit(lsq_ps, name=("norm_beam1", "norm_beam2", "norm_beam3", "const"), *ini_params)
                obj_minuit.limits = [(0,1.0), (0,1.0), (0,1.0), (-100,100)]
                obj_minuit.migrad()
                obj_minuit.hesse()
                chi2dof = obj_minuit.fval / ndof
                str_chi2 = f"𝜒²/ndof = {obj_minuit.fval:.2f} / {ndof} = {chi2dof}"
                print(str_chi2)
                if chi2dof > 1.065:
                    self.do_5_params = True
                else:
                    self.do_4_params = True
            else:
                self.do_3_params = True

        if self.do_3_params == True:
            print(f'2 parameter fitting is not enough, try 3 parameter fitting...')
            ctr2_iflux, ctr2_lon, ctr2_lat = near[0], near[1], near[2]
            print(f'{ctr2_iflux=}, {ctr2_lon=}, {ctr2_lat=}')

            ctr2_vec = hp.ang2vec(theta=ctr2_lon, phi=ctr2_lat, lonlat=True)
            ctr2_ipix_disc = hp.query_disc(nside=self.nside, vec=ctr2_vec, radius=ps_pos_bias * np.deg2rad(self.beam) / 60)
            # chi2dof_arr = np.zeros((len(ctr1_ipix_disc), len(ctr2_ipix_disc)))
            chi2dof_arr = np.full((len(ctr1_ipix_disc), len(ctr2_ipix_disc)), np.nan)
            print(f'{chi2dof_arr.shape}')

            for idx_2, ipix_2 in enumerate(ctr2_ipix_disc):
                for idx_1, ipix_1 in enumerate(ctr1_ipix_disc):
                    self.ctr_ipix_disc = (ipix_1, ipix_2)
                    ini_params = (self.ini_norm_beam, ctr2_iflux, 0.0)
                    obj_minuit = Minuit(lsq_ps, name=("norm_beam1", "norm_beam2", "const"), *ini_params)
                    obj_minuit.limits = [(0,1.0), (0,1.0), (-100,100)]
                    obj_minuit.migrad()
                    obj_minuit.hesse()
                    print(f'{obj_minuit.values}')
                    print(f'{obj_minuit.errors}')
                    chi2dof = obj_minuit.fval / ndof
                    str_chi2 = f"𝜒²/ndof = {obj_minuit.fval:.2f} / {ndof} = {chi2dof}"
                    print(str_chi2)
                    chi2dof_arr[idx_1][idx_2] = chi2dof
                if np.nanmin(chi2dof_arr[:,idx_2]) < 1.065:
                    break

            print(f'{chi2dof_arr=}')
            min_chi2dof = np.nanmin(chi2dof_arr)
            min_chi2dof_idx = np.unravel_index(np.nanargmin(chi2dof_arr), chi2dof_arr.shape)

            print(f'minimum idx is {min_chi2dof_idx}, chi2dof is {min_chi2dof}')

            ipix_1 = ctr1_ipix_disc[min_chi2dof_idx[0]]
            print(f'{ipix_1=}')
            ipix_2 = ctr2_ipix_disc[min_chi2dof_idx[1]]
            print(f'{ipix_2=}')

            self.ctr_ipix_disc = (ipix_1, ipix_2)

            obj_minuit = Minuit(lsq_ps, name=("norm_beam1", "norm_beam2", "const"), *ini_params)
            obj_minuit.limits = [(0,1.0), (0,1.0), (-100,100)]
 
            print(obj_minuit.migrad())
            print(obj_minuit.hesse())
            chi2dof = obj_minuit.fval / ndof
            str_chi2 = f"𝜒²/ndof = {obj_minuit.fval:.2f} / {ndof} = {chi2dof}"
            print(str_chi2)

            is_ps = obj_minuit.values['norm_beam1'] > self.sigma_threshold * obj_minuit.errors['norm_beam1']

            true_norm_beam = self.flux2norm_beam(self.iflux)
            print(f'{true_norm_beam=}')
            print(f'{ctr2_iflux=}')

            if obj_minuit.fmin.hesse_failed:
                raise ValueError('hesse failed !!!')

            if chi2dof < 1.065:
                if is_ps:
                    print(f'3 parameter fitting is enough, hesse ok!')
                    self.fit_norm_beam = obj_minuit.values['norm_beam1']
                    fit_lon, fit_lat = np.deg2rad(hp.pix2ang(nside=self.nside, ipix=ipix_1, lonlat=True))
                    print(f'{self.ini_norm_beam=}')
                    print(f'{self.fit_norm_beam=}')
                    fit_error = np.abs((self.fit_norm_beam - self.ini_norm_beam) / self.ini_norm_beam)
                    print(f'{fit_error=}')
                    return self.fit_norm_beam, fit_lon, fit_lat
                else:
                    print(f'not a point source under {self.sigma_threshold} sigma threshold')
                    return 0.0, np.deg2rad(self.lon), np.deg2rad(self.lat)
            else:
                self.do_4_params = True

        if self.do_4_params == True:
            print(f'3 parameter fitting is not enough, try 4 parameter fitting...')
            ctr2_iflux, ctr2_lon, ctr2_lat, ctr3_iflux, ctr3_lon, ctr3_lat= near[0], near[1], near[2], near[3], near[4], near[5]
            print(f'{ctr2_iflux=}, {ctr2_lon=}, {ctr2_lat=}, {ctr3_iflux=}, {ctr3_lon=}, {ctr3_lat=}')

            ctr2_vec = hp.ang2vec(theta=ctr2_lon, phi=ctr2_lat, lonlat=True)
            ctr2_ipix_disc = hp.query_disc(nside=self.nside, vec=ctr2_vec, radius=ps_pos_bias * np.deg2rad(self.beam) / 60)
            ctr3_vec = hp.ang2vec(theta=ctr3_lon, phi=ctr3_lat, lonlat=True)
            ctr3_ipix_disc = hp.query_disc(nside=self.nside, vec=ctr3_vec, radius=ps_pos_bias * np.deg2rad(self.beam) / 60)


            chi2dof_arr = np.full((len(ctr1_ipix_disc), len(ctr2_ipix_disc), len(ctr3_ipix_disc)), np.nan)
            print(f'{chi2dof_arr.shape}')

            def loop_function_4_params():
                for idx_3, ipix_3 in enumerate(ctr3_ipix_disc):
                    for idx_2, ipix_2 in enumerate(ctr2_ipix_disc):
                        for idx_1, ipix_1 in enumerate(ctr1_ipix_disc):
                            self.ctr_ipix_disc = (ipix_1, ipix_2, ipix_3)
                            print(f'{self.ctr_ipix_disc=}')
                            ini_params = (self.ini_norm_beam, ctr2_iflux, ctr3_iflux, 0.0)
                            obj_minuit = Minuit(lsq_ps, name=("norm_beam1", "norm_beam2", "norm_beam3", "const"), *ini_params)
                            obj_minuit.limits = [(0,1.0), (0,1.0), (0,1.0), (-100,100)]
                            obj_minuit.migrad()
                            obj_minuit.hesse()
                            print(f'{obj_minuit.values}')
                            print(f'{obj_minuit.errors}')
                            chi2dof = obj_minuit.fval / ndof
                            str_chi2 = f"𝜒²/ndof = {obj_minuit.fval:.2f} / {ndof} = {chi2dof}"
                            print(str_chi2)
                            chi2dof_arr[idx_1][idx_2][idx_3] = chi2dof
                        print(f'{chi2dof_arr[:,idx_2,idx_3]=}')
                        if np.nanmin(chi2dof_arr[:,idx_2,idx_3]) < 1.065:
                            return chi2dof_arr
                else:
                    return chi2dof_arr

            chi2dof_arr = loop_function_4_params()
            # print(f'{chi2dof_arr=}')
            min_chi2dof = np.nanmin(chi2dof_arr)
            min_chi2dof_idx = np.unravel_index(np.nanargmin(chi2dof_arr), chi2dof_arr.shape)

            print(f'minimum idx is {min_chi2dof_idx}, chi2dof is {min_chi2dof}')

            ipix_1 = ctr1_ipix_disc[min_chi2dof_idx[0]]
            print(f'{ipix_1=}')
            ipix_2 = ctr2_ipix_disc[min_chi2dof_idx[1]]
            print(f'{ipix_2=}')
            ipix_3 = ctr3_ipix_disc[min_chi2dof_idx[2]]
            print(f'{ipix_3=}')

            self.ctr_ipix_disc = (ipix_1, ipix_2, ipix_3)
            ini_params = (self.ini_norm_beam, ctr2_iflux, ctr3_iflux, 0.0)
            obj_minuit = Minuit(lsq_ps, name=("norm_beam1", "norm_beam2", "norm_beam3", "const"), *ini_params)
            obj_minuit.limits = [(0,1.0), (0,1.0), (0,1.0), (-100,100)]

            print(obj_minuit.migrad())
            print(obj_minuit.hesse())
            chi2dof = obj_minuit.fval / ndof
            str_chi2 = f"𝜒²/ndof = {obj_minuit.fval:.2f} / {ndof} = {chi2dof}"
            print(str_chi2)

            is_ps = obj_minuit.values['norm_beam1'] > self.sigma_threshold * obj_minuit.errors['norm_beam1']

            if obj_minuit.fmin.hesse_failed:
                raise ValueError('hesse failed !!!')

            if chi2dof < 1.065:
                if is_ps:
                    print(f'4 parameter fitting is enough, hesse ok!')
                    self.fit_norm_beam = obj_minuit.values['norm_beam1']
                    fit_lon, fit_lat = np.deg2rad(hp.pix2ang(nside=self.nside, ipix=ipix_1, lonlat=True))
                    print(f'{self.ini_norm_beam=}')
                    print(f'{self.fit_norm_beam=}')
                    fit_error = np.abs((self.fit_norm_beam - self.ini_norm_beam) / self.ini_norm_beam)
                    print(f'{fit_error=}')
                    return self.fit_norm_beam, fit_lon, fit_lat
                else:
                    print(f'not a point source under {self.sigma_threshold} sigma threshold')
                    return 0.0, np.deg2rad(self.lon), np.deg2rad(self.lat)
            else:
                # raise ValueError('4 parameter fitting is not enough, try something else!')
                self.do_5_params = True

        if self.do_5_params == True:
            print(f'4 parameter fitting is not enough, try 5 parameter fitting...')
            ctr2_iflux, ctr2_lon, ctr2_lat, ctr3_iflux, ctr3_lon, ctr3_lat, ctr4_iflux, ctr4_lon, ctr4_lat = near[0], near[1], near[2], near[3], near[4], near[5], near[6], near[7], near[8]
            print(f'{ctr2_iflux=}, {ctr2_lon=}, {ctr2_lat=}, {ctr3_iflux=}, {ctr3_lon=}, {ctr3_lat=}, {ctr4_iflux=}, {ctr4_lon=}, {ctr4_lat=}')

            ctr2_vec = hp.ang2vec(theta=ctr2_lon, phi=ctr2_lat, lonlat=True)
            ctr2_ipix_disc = hp.query_disc(nside=self.nside, vec=ctr2_vec, radius=ps_pos_bias * np.deg2rad(self.beam) / 60)
            ctr3_vec = hp.ang2vec(theta=ctr3_lon, phi=ctr3_lat, lonlat=True)
            ctr3_ipix_disc = hp.query_disc(nside=self.nside, vec=ctr3_vec, radius=ps_pos_bias * np.deg2rad(self.beam) / 60)
            ctr4_vec = hp.ang2vec(theta=ctr4_lon, phi=ctr4_lat, lonlat=True)
            ctr4_ipix_disc = hp.query_disc(nside=self.nside, vec=ctr4_vec, radius=ps_pos_bias * np.deg2rad(self.beam) / 60)

            chi2dof_arr = np.full((len(ctr1_ipix_disc), len(ctr2_ipix_disc), len(ctr3_ipix_disc), len(ctr4_ipix_disc)), np.nan)
            print(f'{chi2dof_arr.shape}')

            def loop_function_5_params():
                for idx_4, ipix_4 in enumerate(ctr4_ipix_disc):
                    for idx_3, ipix_3 in enumerate(ctr3_ipix_disc):
                        for idx_2, ipix_2 in enumerate(ctr2_ipix_disc):
                            for idx_1, ipix_1 in enumerate(ctr1_ipix_disc):
                                self.ctr_ipix_disc = (ipix_1, ipix_2, ipix_3, ipix_4)
                                ini_params = (self.ini_norm_beam, ctr2_iflux, ctr3_iflux, ctr4_iflux, 0.0)
                                obj_minuit = Minuit(lsq_ps, name=("norm_beam1", "norm_beam2", "norm_beam3", "norm_beam4", "const"), *ini_params)
                                obj_minuit.limits = [(0,1.0), (0,1.0), (0,1.0), (0,1.0), (-100,100)]
                                obj_minuit.migrad()
                                obj_minuit.hesse()
                                print(f'{obj_minuit.values}')
                                print(f'{obj_minuit.errors}')
                                chi2dof = obj_minuit.fval / ndof
                                str_chi2 = f"𝜒²/ndof = {obj_minuit.fval:.2f} / {ndof} = {chi2dof}"
                                print(str_chi2)
                                chi2dof_arr[idx_1][idx_2][idx_3][idx_4] = chi2dof

                            if np.nanmin(chi2dof_arr[:,idx_2,idx_3,idx_4]) < 1.065:
                                return chi2dof_arr
                else:
                    return chi2dof_arr

            chi2dof_arr = loop_function_5_params()
            # print(f'{chi2dof_arr=}')
            min_chi2dof = np.nanmin(chi2dof_arr)
            min_chi2dof_idx = np.unravel_index(np.nanargmin(chi2dof_arr), chi2dof_arr.shape)

            print(f'minimum idx is {min_chi2dof_idx}, chi2dof is {min_chi2dof}')

            ipix_1 = ctr1_ipix_disc[min_chi2dof_idx[0]]
            print(f'{ipix_1=}')
            ipix_2 = ctr2_ipix_disc[min_chi2dof_idx[1]]
            print(f'{ipix_2=}')
            ipix_3 = ctr3_ipix_disc[min_chi2dof_idx[2]]
            print(f'{ipix_3=}')
            ipix_4 = ctr4_ipix_disc[min_chi2dof_idx[3]]
            print(f'{ipix_4=}')

            self.ctr_ipix_disc = (ipix_1, ipix_2, ipix_3, ipix_4)
            ini_params = (self.ini_norm_beam, ctr2_iflux, ctr3_iflux, ctr4_iflux, 0.0)

            obj_minuit = Minuit(lsq_ps, name=("norm_beam1", "norm_beam2", "norm_beam3", "norm_beam4", "const"), *ini_params)
            obj_minuit.limits = [(0,1.0), (0,1.0), (0,1.0), (0,1.0), (-100,100)]

            print(obj_minuit.migrad())
            print(obj_minuit.hesse())
            chi2dof = obj_minuit.fval / ndof
            str_chi2 = f"𝜒²/ndof = {obj_minuit.fval:.2f} / {ndof} = {chi2dof}"
            print(str_chi2)

            is_ps = obj_minuit.values['norm_beam1'] > self.sigma_threshold * obj_minuit.errors['norm_beam1']

            if obj_minuit.fmin.hesse_failed:
                raise ValueError('hesse failed !!!')

            if chi2dof < 1.065:
                if is_ps:
                    print(f'4 parameter fitting is enough, hesse ok!')
                    self.fit_norm_beam = obj_minuit.values['norm_beam1']
                    fit_lon, fit_lat = np.deg2rad(hp.pix2ang(nside=self.nside, ipix=ipix_1, lonlat=True))
                    print(f'{self.ini_norm_beam=}')
                    print(f'{self.fit_norm_beam=}')
                    fit_error = np.abs((self.fit_norm_beam - self.ini_norm_beam) / self.ini_norm_beam)
                    print(f'{fit_error=}')
                    return self.fit_norm_beam, fit_lon, fit_lat
                else:
                    print(f'not a point source under {self.sigma_threshold} sigma threshold')
                    return 0.0, np.deg2rad(self.lon), np.deg2rad(self.lat)
            else:
                raise ValueError('5 parameter fitting is not enough, try something else!')









if __name__ == '__main__':
    m = np.load('../../FGSim/PSNOISE/2048/40.npy')[0]
    nstd = np.load('../../FGSim/NSTDNORTH/2048/40.npy')[0]
    df_mask = pd.read_csv('../partial_sky_ps/ps_in_mask/mask40.csv')
    flux_idx = 53
    lon = np.rad2deg(df_mask.at[flux_idx, 'lon'])
    lat = np.rad2deg(df_mask.at[flux_idx, 'lat'])
    iflux = df_mask.at[flux_idx, 'iflux']

    df_ps = pd.read_csv('../../test/ps_sort/sort_by_iflux/40.csv')
    
    lmax = 350
    nside = 2048
    beam = 63
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax)
    cl_cmb = np.load('../../src/cmbsim/cmbdata/cmbcl.npy')[:lmax+1,0]
    cl_cmb = cl_cmb * bl**2

    obj = FitPointSource(m=m, nstd=nstd, flux_idx=flux_idx, df_mask=df_mask, df_ps=df_ps, cl_cmb=cl_cmb, lon=lon, lat=lat, iflux=iflux, lmax=lmax, nside=nside, radius_factor=1.0, beam=beam)

    # obj.see_true_map(m=m, lon=lon, lat=lat, nside=nside, beam=beam)
    obj.fit_ps_ns()

