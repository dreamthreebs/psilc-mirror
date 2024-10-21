import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from pathlib import Path
from eblc_base_slope import EBLeakageCorrection

noise_seed = np.load('../../seeds_noise_2k.npy')
cmb_seed = np.load('../../seeds_cmb_2k.npy')
fg_seed = np.load('../../seeds_fg_2k.npy')

rlz_idx=0
nside = 2048
npix = hp.nside2npix(nside=nside)
beam = 67

def gen_fg_cl():
    Cl_TT = np.load('../../Cl_fg/data/cl_fg_TT.npy')
    Cl_EE = np.load('../../Cl_fg/data/cl_fg_EE.npy')
    Cl_BB = np.load('../../Cl_fg/data/cl_fg_BB.npy')
    Cl_TE = np.zeros_like(Cl_TT)
    return np.array([Cl_TT, Cl_EE, Cl_BB, Cl_TE])

def gen_map():

    ps = np.load('../data/ps/ps.npy')

    nstd = np.load('../../../FGSim/NSTDNORTH/2048/30.npy')
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

    cls_fg = gen_fg_cl()
    np.random.seed(seed=fg_seed[rlz_idx])
    fg_iqu = hp.synfast(cls_fg, nside=nside, fwhm=0, new=True, lmax=600)


    m = noise + ps + cmb_iqu + fg_iqu
    # m = noise
    # cn = noise + cmb_iqu

    # m = np.load('./1_8k.npy')
    # np.save('./1_6k_pcn.npy', m)
    # np.save('./1_6k_cn.npy', cn)
    return m, noise

m_pcfn, m_n = gen_map()
# m_b = hp.alm2map(hp.map2alm(m)[2], nside=2048)

# path_m = Path(f'./n')
# path_m.mkdir(exist_ok=True, parents=True)

# hp.write_map(path_m / Path(f'{rlz_idx}.npy'), m_b)
slope_in = np.load(f'./eblc_slope/{rlz_idx}.npy')
mask = hp.read_map(f'./mask/mask.fits')

obj = EBLeakageCorrection(m_pcfn, lmax=3*nside-1, nside=nside, mask=mask, post_mask=mask, slope_in=slope_in)
_,_,cln_b_pcfn = obj.run_eblc()

obj = EBLeakageCorrection(m_n, lmax=3*nside-1, nside=nside, mask=mask, post_mask=mask, slope_in=slope_in)
_,_,cln_b_n = obj.run_eblc()

path_input = Path('./input')
path_input.mkdir(exist_ok=True, parents=True)
hp.write_map(path_input / Path(f'{rlz_idx}.fits'), cln_b_pcfn, overwrite=True)

path_input = Path('./input_n')
path_input.mkdir(exist_ok=True, parents=True)
hp.write_map(path_input / Path(f'{rlz_idx}.fits'), cln_b_n, overwrite=True)






