import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt
import logging

from iminuit import Minuit
from pathlib import Path
from fit_b_v2 import Fit_on_B

# logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s -%(name)s - %(message)s')
logging.basicConfig(level=logging.WARNING)
# logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)

def check_one_ps():

    df_mask = pd.read_csv('../../pp_P/mask/mask_csv/215.csv')
    df_ps = pd.read_csv('../../pp_P/mask/ps_csv/215.csv')
    lmax = 1999
    nside = 2048
    beam = 11
    freq = 215
    flux_idx = 7
    lon = df_mask.at[flux_idx, 'lon']
    print(f'{lon=}')
    lat = df_mask.at[flux_idx, 'lat']
    qflux = df_mask.at[flux_idx, 'qflux']
    uflux = df_mask.at[flux_idx, 'uflux']
    pflux = df_mask.at[flux_idx, 'pflux']

    print(f'{lon=}, {lat=}, {qflux=}, {uflux=}, {pflux=}')
    print(df_mask.head())
    # m = np.load('../../fitdata/synthesis_data/2048/PSCMBNOISE/215/1.npy').copy()
    # m = np.load('../../fitdata/2048/PS/215/ps.npy').copy()
    # m_b = hp.alm2map(hp.map2alm(m)[2], nside=nside)
    # np.save(f'./{flux_idx}.npy', m_b)
    # m_b = np.load('./1.npy')
    m_b = np.load('./1_6k_pcn.npy')


    obj = Fit_on_B(m_b, df_mask, df_ps, flux_idx, qflux, uflux, pflux, lmax, nside, beam, lon, lat, freq, r_fold=2.5, r_fold_rmv=5)
    # obj.check_ps()
    obj.params_for_fitting()
    num_ps = obj.test_number_nearby_ps(threshold_extra_factor=9.5)
    print(f'{num_ps=}')
    # obj.see_true_map(nside=nside, beam=beam)
    # obj.see_b_map(nside, beam)
    # obj.calc_inv_cov(mode='n1')
    # obj.calc_inv_cov(mode='cn1')
    # obj.ez_fit_b()
    # obj.test_fit_b()
    # obj.params_for_testing()
    # obj.test_residual()

def check_all_ps():

    df_mask = pd.read_csv('../../pp_P/mask/mask_csv/215.csv')
    df_ps = pd.read_csv('../../pp_P/mask/ps_csv/215.csv')
    lmax = 1999
    nside = 2048
    beam = 11
    freq = 215

    m_b = np.load('./1_6k_pcn.npy')

    num_ps_list = []
    for flux_idx in range(213):
        lon = df_mask.at[flux_idx, 'lon']
        print(f'{lon=}')
        lat = df_mask.at[flux_idx, 'lat']
        qflux = df_mask.at[flux_idx, 'qflux']
        uflux = df_mask.at[flux_idx, 'uflux']
        pflux = df_mask.at[flux_idx, 'pflux']

        print(f'{lon=}, {lat=}, {qflux=}, {uflux=}, {pflux=}')

        obj = Fit_on_B(m_b, df_mask, df_ps, flux_idx, qflux, uflux, pflux, lmax, nside, beam, lon, lat, freq, r_fold=2.5, r_fold_rmv=5)
        # obj.check_ps()
        obj.params_for_fitting()
        num_ps = obj.test_number_nearby_ps(threshold_extra_factor=9.5)
        print(f'{num_ps=}')
        num_ps_list.append(num_ps)
        # obj.see_true_map(nside=nside, beam=beam)
        # obj.see_b_map(nside, beam)
        # obj.calc_inv_cov(mode='n1')
        # obj.calc_inv_cov(mode='cn1')
        # obj.ez_fit_b()
        # obj.test_fit_b()
        # obj.params_for_testing()
        # obj.test_residual()

    print(f'{num_ps_list}')

# check_one_ps()
check_all_ps()



