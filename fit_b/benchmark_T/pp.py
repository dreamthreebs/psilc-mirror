import numpy as np
import healpy as hp
import pandas as pd
import logging
import matplotlib.pyplot as plt

from fit_b import Fit_on_B
from pathlib import Path
# logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s -%(name)s - %(message)s')
logging.basicConfig(level=logging.WARNING)
# logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)

def main():
    df_mask = pd.read_csv('./mask/30.csv')
    df_ps = pd.read_csv('../../pp_P/mask/ps_csv/30.csv')
    lmax = 1999
    nside = 2048
    beam = 67
    freq = 30
    flux_idx = 0

    # m_b = np.load('./data/ps/ps_b.npy')
    # noise = 0.1 * np.random.normal(loc=0, scale=1, size=(hp.nside2npix(nside),))
    # m_b = m_b + noise

    m_b = np.load('./data/pcn_b.npy')

    for flux_idx in range(10):
        lon = df_mask.at[flux_idx, 'lon']
        print(f'{lon=}')
        lat = df_mask.at[flux_idx, 'lat']
        qflux = df_mask.at[flux_idx, 'qflux']
        uflux = df_mask.at[flux_idx, 'uflux']
        pflux = df_mask.at[flux_idx, 'pflux']

        obj = Fit_on_B(m_b, df_mask, df_ps, flux_idx, qflux, uflux, pflux, lmax, nside, beam, lon, lat, freq, r_fold=2.5, r_fold_rmv=5)
        obj.params_for_fitting()

main()

