import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt

from fit_b import Fit_on_B
from pathlib import Path

lmax = 2000
nside = 2048
npix = hp.nside2npix(nside)
beam = 11
rlz_idx=0

flux_idx = 1


def gen_b_map():

    ps = np.load('../../fitdata/2048/PS/215/ps.npy')

    nstd = np.load('../../FGSim/NSTDNORTH/2048/215.npy')
    noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))
    print(f"{np.std(noise[1])=}")

    # cmb_iqu = np.load(f'../../fitdata/2048/CMB/215/{rlz_idx}.npy')
    # cls = np.load('../../src/cmbsim/cmbdata/cmbcl.npy')
    cls = np.load('../../src/cmbsim/cmbdata/cmbcl_8k.npy')
    # cmb_iqu = hp.synfast(cls.T, nside=nside, fwhm=np.deg2rad(beam)/60, new=True, lmax=1999)
    cmb_iqu = hp.synfast(cls.T, nside=nside, fwhm=np.deg2rad(beam)/60, new=True, lmax=3*nside-1)

    # l = np.arange(lmax+1)
    # cls_out = hp.anafast(cmb_iqu, lmax=lmax)

    # plt.loglog(l, l*(l+1)*cls_out[2])
    # plt.show()


    # pcn = ps + cmb_iqu
    # pcn = ps + noise

    m = noise + ps + cmb_iqu
    # cn = noise + cmb_iqu
    m_b = hp.alm2map(hp.map2alm(m)[2], nside=nside)
    # m_b_cn = hp.alm2map(hp.map2alm(cn)[2], nside=nside)

    # m = np.load('./1_8k.npy')
    # np.save('./1_6k_pcn.npy', m_b)
    # np.save('./1_6k_cn.npy', m_b_cn)
    return m_b

def main():
    m_b = gen_b_map()

    df = pd.read_csv('../../pp_P/mask/mask_csv/215.csv')
    lmax = 1999
    nside = 2048
    beam = 11
    freq = 215
    flux_idx = 1
    lon = df.at[flux_idx, 'lon']
    print(f'{lon=}')
    lat = df.at[flux_idx, 'lat']
    qflux = df.at[flux_idx, 'qflux']
    uflux = df.at[flux_idx, 'uflux']
    pflux = df.at[flux_idx, 'pflux']

    obj = Fit_on_B(m_b, flux_idx, qflux, uflux, pflux, lmax, nside, beam, lon, lat, freq, r_fold=2.5, r_fold_rmv=5)
    obj.params_for_fitting()
    # obj.calc_inv_cov(mode='cn')
    obj.calc_inv_cov(mode='cn1')
    # obj.calc_inv_cov(mode='n1')
    obj.fit_b()
    print(f'{obj.P=}, {obj.ps_2phi=}')

    path_params = Path('./params')
    path_params.mkdir(parents=True, exist_ok=True)

    np.save(path_params / Path(f'P_{rlz_idx}.npy'), obj.P)
    np.save(path_params / Path(f'phi_{rlz_idx}.npy'), obj.ps_2phi)

    # obj.params_for_testing()
    # obj.test_residual()


main()

