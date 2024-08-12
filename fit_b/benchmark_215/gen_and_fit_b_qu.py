import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pandas as pd
import time
import pickle
import os,sys
import logging
import ipdb

from pathlib import Path
from iminuit import Minuit
from iminuit.cost import LeastSquares
from fit_b import Fit_on_B

# logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s -%(name)s - %(message)s')
logging.basicConfig(level=logging.WARNING)
# logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.INFO)

rlz_idx=0
nside = 2048
lmax = 3 * nside - 1
npix = hp.nside2npix(nside)
beam = 11

noise_seed = np.load('./seeds_noise_2k.npy')
cmb_seed = np.load('./seeds_cmb_2k.npy')

def gen_b_map():

    ps = np.load('./data/ps/ps.npy')

    nstd = np.load('../../FGSim/NSTDNORTH/2048/215.npy')
    np.random.seed(seed=noise_seed[rlz_idx])
    noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))
    print(f"{np.std(noise[1])=}")

    # cmb_iqu = np.load(f'../../fitdata/2048/CMB/215/{rlz_idx}.npy')
    # cls = np.load('../../src/cmbsim/cmbdata/cmbcl.npy')
    cls = np.load('../../src/cmbsim/cmbdata/cmbcl_8k.npy')
    np.random.seed(seed=cmb_seed[rlz_idx])
    # cmb_iqu = hp.synfast(cls.T, nside=nside, fwhm=np.deg2rad(beam)/60, new=True, lmax=1999)
    cmb_iqu = hp.synfast(cls.T, nside=nside, fwhm=np.deg2rad(beam)/60, new=True, lmax=3*nside-1)

    # l = np.arange(lmax+1)
    # cls_out = hp.anafast(cmb_iqu, lmax=lmax)

    # plt.loglog(l, l*(l+1)*cls_out[2])
    # plt.show()


    # pcn = ps + cmb_iqu
    # pcn = ps + noise

    m = noise + ps + cmb_iqu
    m_b = hp.alm2map(hp.map2alm(m)[2], nside=nside)
    # cn = noise + cmb_iqu

    # m = np.load('./1_8k.npy')
    # np.save('./1_6k_pcn.npy', m)
    # np.save('./1_6k_cn.npy', cn)
    return m_b


def main():
    m_b = gen_b_map()

    df_mask = pd.read_csv('./mask/215.csv')
    df_ps = pd.read_csv('../../pp_P/mask/ps_csv/215.csv')
    lmax = 1999
    nside = 2048
    beam = 11
    freq = 215
    flux_idx = 0
    lon = df_mask.at[flux_idx, 'lon']
    print(f'{lon=}')
    lat = df_mask.at[flux_idx, 'lat']
    qflux = df_mask.at[flux_idx, 'qflux']
    uflux = df_mask.at[flux_idx, 'uflux']
    pflux = df_mask.at[flux_idx, 'pflux']

    obj = Fit_on_B(m_b, df_mask, df_ps, flux_idx, qflux, uflux, pflux, lmax, nside, beam, lon, lat, freq, r_fold=2.5, r_fold_rmv=5)

    obj.params_for_fitting()
    obj.calc_inv_cov(mode='cn2')
    obj.fit_1_ps()



main()

