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

from fit_qu_base import FitPolPS

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.INFO)

freq = 215
lmax = 1999
nside = 2048
beam = 11

def filter_df():
    # filter pandas dataframe where flux bigger than some threshold
    df_mask = pd.read_csv(f'../mask/mask_csv/{freq}.csv')
    filtered_df = df_mask[df_mask['pflux']>1]
    logger.info(f'{filtered_df=}')
    logger.info(f'{len(filtered_df)=}')

def calc_number():
    # calculate number of point source near one point source

    time0 = time.perf_counter()
    # m = np.load(f'../../fitdata/synthesis_data/2048/PSNOISE/{freq}/0.npy')
    m = np.load(f'../../fitdata/synthesis_data/2048/PSCMBNOISE/{freq}/2.npy')
    m_q = m[1].copy()
    m_u = m[2].copy()
    logger.debug(f'{sys.getrefcount(m_q)-1=}')

    logger.info(f'time for fitting = {time.perf_counter()-time0}')
    nstd = np.load(f'../../FGSim/NSTDNORTH/2048/{freq}.npy')
    nstd_q = nstd[1].copy()
    nstd_u = nstd[2].copy()
    df_mask = pd.read_csv(f'../mask/mask_csv/{freq}.csv')
    df_ps = pd.read_csv(f'../mask/ps_csv/{freq}.csv')

    num_ps_list = []
    overlap_list = []
    for flux_idx in range(700):

        logger.debug(f'{sys.getrefcount(m_q)-1=}')
        obj = FitPolPS(m_q=m_q, m_u=m_u, freq=freq, nstd_q=nstd_q, nstd_u=nstd_u, flux_idx=flux_idx, df_mask=df_mask, df_ps=df_ps, lmax=lmax, nside=nside, radius_factor=1.5, beam=beam, epsilon=0.00001)

        logger.debug(f'{sys.getrefcount(m_q)-1=}')
        # obj.see_true_map(m_q=m_q, m_u=m_u, nside=nside, beam=beam)

        num_ps, num_near_ps, ang_near = obj.fit_all(cov_mode='noise', mode='get_num_ps')
        overlap_flag = obj.fit_all(cov_mode='noise', mode='get_overlap')
        logger.debug(f'{num_ps=}, {num_near_ps=}, {ang_near}, {overlap_flag}')
        num_ps_list.append(num_ps)
        if overlap_flag == True:
            overlap_list.append(flux_idx)

    num_ps_arr = np.asarray(num_ps_list)
    overlap_arr = np.asarray(overlap_list)
    print(f'{num_ps_arr=}')
    print(f'{overlap_arr=}')

def get_disc_pix_ind():
    # get pixel index for every point sources in catelogue

    time0 = time.perf_counter()
    # m = np.load(f'../../fitdata/synthesis_data/2048/PSNOISE/{freq}/0.npy')
    m = np.load(f'../../fitdata/synthesis_data/2048/PSCMBNOISE/{freq}/2.npy')
    m_q = m[1].copy()
    m_u = m[2].copy()
    logger.debug(f'{sys.getrefcount(m_q)-1=}')

    logger.info(f'time for fitting = {time.perf_counter()-time0}')
    nstd = np.load(f'../../FGSim/NSTDNORTH/2048/{freq}.npy')
    nstd_q = nstd[1].copy()
    nstd_u = nstd[2].copy()
    df_mask = pd.read_csv(f'../mask/mask_csv/{freq}.csv')
    df_ps = pd.read_csv(f'../mask/ps_csv/{freq}.csv')

    for flux_idx in range(214):
        obj = FitPolPS(m_q=m_q, m_u=m_u, freq=freq, nstd_q=nstd_q, nstd_u=nstd_u, flux_idx=flux_idx, df_mask=df_mask, df_ps=df_ps, lmax=lmax, nside=nside, radius_factor=1.5, beam=beam, epsilon=0.00001)
        pix_ind = obj.get_pix_ind()
        path_pix_ind = Path('./pix_idx')
        path_pix_ind.mkdir(exist_ok=True, parents=True)
        np.save(path_pix_ind / Path(f'./{flux_idx}.npy'), pix_ind)
        print(f'{pix_ind=}')

# filter_df()
# calc_number()
get_disc_pix_ind()


