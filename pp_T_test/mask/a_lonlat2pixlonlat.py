import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pandas as pd
import time
import pickle
import os
import logging

from pathlib import Path

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
# logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LonLat2PixLonLat:
    def __init__(self, df_mask, df_ps, nside, freq):
        self.df_mask = df_mask # pandas data frame of point sources in mask
        self.df_ps = df_ps # pandas data frame of all point sources
        self.nside = nside # Assuming nside is provided during class instantiation
        self.freq = freq

    def input_lonlat2pix_lonlat(self, input_lon, input_lat):
        logger.debug('{input_lon=}, {input_lat=}')
        ipix = hp.ang2pix(nside=self.nside, theta=input_lon, phi=input_lat, lonlat=True)
        out_lon, out_lat = hp.pix2ang(nside=self.nside, ipix=ipix, lonlat=True)
        return out_lon, out_lat

    def change_lonlat2pixlonlat(self):
        logger.info(f'Changing lon lat in Catalogue to nside 2048 pixel lon lat...')
        lon = np.rad2deg(self.df_ps.loc[:, 'lon'].to_numpy())
        lat = np.rad2deg(self.df_ps.loc[:, 'lat'].to_numpy())
        logger.debug(f'{lon.shape=}')
        logger.debug(f'{lat.shape=}')
        ipix = hp.ang2pix(nside=self.nside, theta=lon, phi=lat, lonlat=True)
        pix_lon, pix_lat = hp.pix2ang(nside=self.nside, ipix=ipix, lonlat=True)
        pix_lon_rad = np.deg2rad(pix_lon)
        pix_lat_rad = np.deg2rad(pix_lat)

        diff_lon = pix_lon - lon
        diff_lat = pix_lat - lat
        logger.debug(f'{diff_lon=}')
        logger.debug(f'{diff_lat=}')
        logger.debug(f'{np.max(np.abs(diff_lon))=}')
        logger.debug(f'{np.max(np.abs(diff_lat))=}')

        self.df_ps['lon'] = pix_lon_rad
        self.df_ps['lat'] = pix_lat_rad
        
        ps_csv_path = Path('./ps_csv')
        ps_csv_path.mkdir(parents=True, exist_ok=True)
        self.df_ps.to_csv(ps_csv_path / Path(f'./{freq}.csv'), index=False)

if __name__ == '__main__':

    nside = 2048
    # df_mask = pd.read_csv('../../psfit/partial_sky_ps/ps_in_mask/mask40.csv')

    # freq = 40
    # df_mask = None
    # df_ps = pd.read_csv(f'../../test/ps_sort/sort_by_iflux/{freq}.csv')
    # df_ps.rename(columns={df_ps.columns[0]: 'flux_idx'}, inplace=True)
    # obj = LonLat2PixLonLat(df_mask, df_ps, nside=nside, freq=freq)
    # obj.change_lonlat2pixlonlat()

    df_mask = None

    for freq in [30, 40, 85, 95, 145, 155, 215, 270]:
        df_ps = pd.read_csv(f'../../test/ps_sort/sort_by_iflux/{freq}.csv')
        df_ps.rename(columns={df_ps.columns[0]: 'flux_idx'}, inplace=True)

        obj = LonLat2PixLonLat(df_mask, df_ps, nside=nside, freq=freq)

        obj.change_lonlat2pixlonlat()



