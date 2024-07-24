import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt
import logging

from iminuit import Minuit
from fit_b_v2 import Fit_on_B
from pathlib import Path

# logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s -%(name)s - %(message)s')
logging.basicConfig(level=logging.WARNING)
# logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)

noise_seeds = np.load('./seeds_noise_2k.npy')
cmb_seeds = np.load('./seeds_cmb_2k.npy')

def gen_b_map(rlz_idx, nside, npix, beam):

    ps = np.load('../../fitdata/2048/PS/215/ps.npy')

    nstd = np.load('../../FGSim/NSTDNORTH/2048/215.npy')
    np.random.seed(seed=noise_seeds[rlz_idx])
    noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))
    print(f"{np.std(noise[1])=}")

    # cmb_iqu = np.load(f'../../fitdata/2048/CMB/215/{rlz_idx}.npy')
    # cls = np.load('../../src/cmbsim/cmbdata/cmbcl.npy')
    cls = np.load('../../src/cmbsim/cmbdata/cmbcl_8k.npy')
    # cmb_iqu = hp.synfast(cls.T, nside=nside, fwhm=np.deg2rad(beam)/60, new=True, lmax=1999)
    np.random.seed(seed=cmb_seeds[rlz_idx])
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
    # m_b_cn = hp.alm2map(hp.map2alm(cn)[2], nside=nside)

    # m = np.load('./1_8k.npy')
    # np.save('./1_6k_pcn.npy', m_b)
    # np.save('./1_6k_cn.npy', m_b_cn)
    return m_b

if __name__=='__main__':

    rlz_idx=0
    lmax = 1999
    nside = 2048
    beam = 11
    freq = 215
    m_b = gen_b_map(rlz_idx, nside, npix=hp.nside2npix(nside=nside), beam=beam)
    # m_b = np.load('./1_6k_pcn.npy')

    df_mask = pd.read_csv('../../pp_P/mask/mask_csv/215.csv')
    df_ps = pd.read_csv('../../pp_P/mask/ps_csv/215.csv')
    for flux_idx in range(213):

        print(f'{flux_idx=}')
        lon = df_mask.at[flux_idx, 'lon']
        lat = df_mask.at[flux_idx, 'lat']
        qflux = df_mask.at[flux_idx, 'qflux']
        uflux = df_mask.at[flux_idx, 'uflux']
        pflux = df_mask.at[flux_idx, 'pflux']

        print(f'{lon=}, {lat=}, {qflux=}, {uflux=}, {pflux=}')

        obj = Fit_on_B(m_b, df_mask, df_ps, flux_idx, qflux, uflux, pflux, lmax, nside, beam, lon, lat, freq, r_fold=2.5, r_fold_rmv=5)

        obj.params_for_fitting()
        obj.calc_inv_cov(mode='cn1')
        obj.run_fit()

        path_res = Path(f'./fit_res/pcn_params/16core/idx_{flux_idx}')
        path_res.mkdir(exist_ok=True, parents=True)
        print(f"{obj.chi2dof=}, {obj.fit_P=}, {obj.fit_P_err=}, {obj.fit_phi=}, {obj.fit_phi_err=}")
        np.save(path_res / Path(f'chi2dof_{rlz_idx}.npy'), obj.chi2dof)
        np.save(path_res / Path(f'fit_P_{rlz_idx}.npy'), obj.fit_P)
        np.save(path_res / Path(f'fit_P_err_{rlz_idx}.npy'), obj.fit_P_err)
        np.save(path_res / Path(f'fit_phi_{rlz_idx}.npy'), obj.fit_phi)
        np.save(path_res / Path(f'fit_phi_err_{rlz_idx}.npy'), obj.fit_phi_err)

        # obj.test_lsq_params()
        # obj.params_for_testing()
        # obj.test_residual()

