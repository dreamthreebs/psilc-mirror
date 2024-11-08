import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from pathlib import Path

nside = 1024
npix = hp.nside2npix(nside)
beam = 11
freq = 215
rlz_idx = 0

noise_seeds = np.load('../benchmark_215/seeds_noise_2k.npy')
cmb_seeds = np.load('../benchmark_215/seeds_cmb_2k.npy')

path_data = Path(f'./data')
path_data.mkdir(exist_ok=True, parents=True)

def gen_ps_map():
    ipix_ctr = hp.ang2pix(theta=0, phi=0, lonlat=True, nside=nside)
    print(f'{ipix_ctr=}')

    delta_m = np.zeros(npix)
    flux_I = 10000
    delta_m[ipix_ctr] = flux_I

    sm_m = hp.smoothing(delta_m, fwhm=np.deg2rad(beam)/60, pol=False)

    np.save(path_data / Path('ps_map.npy'), sm_m)

    hp.gnomview(sm_m, rot=[0,0,0])
    plt.show()

gen_ps_map()

def gen_cn_map():
    nstd = np.load(f'../../FGSim/NSTDNORTH/1024/{freq}.npy')
    np.random.seed(seed=noise_seeds[rlz_idx])
    noise = nstd[0] * np.random.normal(loc=0, scale=1, size=npix)
    print(f"{np.std(noise)=}")

    cls = np.load('../../src/cmbsim/cmbdata/cmbcl_8k.npy')
    print(f'{cls.T.shape=}')
    np.random.seed(seed=cmb_seeds[rlz_idx])
    # cmb_iqu = hp.synfast(cls.T, nside=nside, fwhm=np.deg2rad(beam)/60, new=True, lmax=1999)
    cmb_i = hp.synfast(cls.T[0], nside=nside, fwhm=np.deg2rad(beam)/60, lmax=3*nside-1)

    np.save(path_data / Path('cmb_i.npy'), cmb_i)
    np.save(path_data / Path('noise_i.npy'), noise)

gen_cn_map()

def gen_pcn_map():
    ps = np.load('./data/ps_map.npy')
    cmb = np.load('./data/cmb_i.npy')
    noise = np.load('./data/noise_i.npy')

    pcn = ps + cmb + noise
    cn = cmb + noise

    np.save(path_data / Path('pcn.npy'), pcn)
    np.save(path_data / Path('cn.npy'), cn)

gen_pcn_map()

