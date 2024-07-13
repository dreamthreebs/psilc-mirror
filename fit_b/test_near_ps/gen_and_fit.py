import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import logging

from fit_b_v1 import Fit_on_B
from pathlib import Path

nside = 2048
beam = 11
npix = hp.nside2npix(nside)

lon1 = 0
lat1 = 0
ipix_ctr1 = hp.ang2pix(theta=lon1, phi=lat1, lonlat=True, nside=nside)
pix_lon1, pix_lat1 = hp.pix2ang(ipix=ipix_ctr1, nside=nside, lonlat=True)

def gen_map():
    ps = np.load('./data/ps/ps_map_12.5.npy')
    noise = 0.1 * np.random.normal(loc=0, scale=1, size=(npix,))
    m = ps + noise
    return m

radius = 12.5

def main():
    m = gen_map()
    # hp.gnomview(m, rot=[pix_lon1, pix_lat1, 0])
    # plt.show()

    obj = Fit_on_B(m=m, lmax=3*nside-1, nside=nside, beam=beam, lon=pix_lon1, lat=pix_lat1)
    
    obj.params_for_fitting()
    obj.calc_inv_cov(mode='n')
    obj.fit_b()

def main_rlz():

    path_bias = Path(f'./data/bias/{radius}')
    path_bias.mkdir(exist_ok=True, parents=True)

    rlz_idx=0
    print(f'{rlz_idx=}')
    m = gen_map()
    # hp.gnomview(m, rot=[pix_lon1, pix_lat1, 0])
    # plt.show()

    obj = Fit_on_B(m=m, lmax=3*nside-1, nside=nside, beam=beam, lon=pix_lon1, lat=pix_lat1)
    obj.params_for_fitting()
    obj.calc_inv_cov(mode='n')
    obj.fit_b()
    P = obj.P
    phi = obj.ps_2phi

    # np.save(f'./data/bias/{radius}/P_{rlz_idx}.npy', P)
    # np.save(f'./data/bias/{radius}/phi_{rlz_idx}.npy', phi)

main_rlz()



