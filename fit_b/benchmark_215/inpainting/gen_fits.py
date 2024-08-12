import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from pathlib import Path

noise_seed = np.load('../seeds_noise_2k.npy')
cmb_seed = np.load('../seeds_cmb_2k.npy')

rlz_idx=0
nside = 2048
npix = hp.nside2npix(nside=nside)
beam = 11
def gen_map():

    ps = np.load('../data/ps/ps.npy')

    nstd = np.load('../../../FGSim/NSTDNORTH/2048/215.npy')
    np.random.seed(seed=noise_seed[rlz_idx])
    # noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))
    noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))
    print(f"{np.std(noise[1])=}")

    # cmb_iqu = np.load(f'../../fitdata/2048/CMB/215/{rlz_idx}.npy')
    # cls = np.load('../../src/cmbsim/cmbdata/cmbcl.npy')
    cls = np.load('../../../src/cmbsim/cmbdata/cmbcl_8k.npy')
    np.random.seed(seed=cmb_seed[rlz_idx])
    # cmb_iqu = hp.synfast(cls.T, nside=nside, fwhm=np.deg2rad(beam)/60, new=True, lmax=1999)
    cmb_iqu = hp.synfast(cls.T, nside=nside, fwhm=np.deg2rad(beam)/60, new=True, lmax=3*nside-1)

    # l = np.arange(lmax+1)
    # cls_out = hp.anafast(cmb_iqu, lmax=lmax)


    m = noise + ps + cmb_iqu
    # cn = noise + cmb_iqu

    # m = np.load('./1_8k.npy')
    # np.save('./1_6k_pcn.npy', m)
    # np.save('./1_6k_cn.npy', cn)
    return m

m = gen_map()
path_input = Path('./input')
path_input.mkdir(exist_ok=True, parents=True)
hp.write_map(path_input / Path(f'q_{rlz_idx}.fits'), m[1], overwrite=True)
hp.write_map(path_input / Path(f'u_{rlz_idx}.fits'), m[2], overwrite=True)

