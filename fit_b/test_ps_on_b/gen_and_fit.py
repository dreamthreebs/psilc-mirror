import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from fit_b_v1 import Fit_on_B
from pathlib import Path

lmax = 2000
nside = 2048
npix = hp.nside2npix(nside)
beam = 11
rlz_idx=0

ctr_ori_lon = 0
ctr_ori_lat = 0

ipix_ctr = hp.ang2pix(theta=ctr_ori_lon, phi=ctr_ori_lat, lonlat=True, nside=nside)
ctr_theta, ctr_phi = hp.pix2ang(nside=nside, ipix=ipix_ctr)
ctr_vec = np.asarray(hp.pix2vec(nside=nside, ipix=ipix_ctr))
ctr_lon, ctr_lat = hp.pix2ang(nside=nside, ipix=ipix_ctr, lonlat=True)
print(f'{ctr_theta=}, {ctr_phi=}, {ctr_vec=}')

def gen_b_map():

    ps = np.load('./data/ps.npy')
    nstd = 0.1
    noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))
    print(f"{np.std(noise[1])=}")

    # cmb_iqu = np.load(f'../../fitdata/2048/CMB/215/{rlz_idx}.npy')
    cls = np.load('../../src/cmbsim/cmbdata/cmbcl.npy')
    cmb_iqu = hp.synfast(cls.T, nside=nside, fwhm=np.deg2rad(beam)/60, new=True, lmax=1999)

    # l = np.arange(lmax+1)
    # cls_out = hp.anafast(cmb_iqu, lmax=lmax)

    # plt.loglog(l, l*(l+1)*cls_out[2])
    # plt.show()

    # pcn = noise + ps + cmb_iqu

    pcn = ps + cmb_iqu
    # pcn = ps + noise
    pcn_b = hp.alm2map(hp.map2alm(pcn)[2], nside=nside)

    # pcn_b = np.load('./m_qu_pcn.npy')
    return pcn_b

def main():
    m = gen_b_map()

    obj = Fit_on_B(m=m, lmax=lmax, nside=nside, beam=beam, lon=ctr_ori_lon, lat=ctr_ori_lat, r_fold=2.5, r_fold_rmv=5)
    obj.params_for_fitting()
    # obj.calc_inv_cov(mode='cn')
    obj.calc_inv_cov(mode='c')
    # obj.calc_inv_cov(mode='n1')
    obj.fit_b()
    print(f'{obj.P=}, {obj.ps_2phi=}')

    path_params = Path('./params')
    path_params.mkdir(parents=True, exist_ok=True)

    # np.save(path_params / Path(f'P_{rlz_idx}.npy'), obj.P)
    # np.save(path_params / Path(f'phi_{rlz_idx}.npy'), obj.ps_2phi)

    obj.params_for_testing()
    obj.test_residual()


main()
