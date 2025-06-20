import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import os,sys

from pathlib import Path
from eblc_base_slope import EBLeakageCorrection
config_dir = Path(__file__).parent.parent
print(f'{config_dir=}')
sys.path.insert(0, str(config_dir))
from config import freq, lmax, nside, beam

rlz_idx=0

noise_seed = np.load('../../seeds_noise_2k.npy')
cmb_seed = np.load('../../seeds_cmb_2k.npy')
fg_seed = np.load('../../seeds_fg_2k.npy')

npix = hp.nside2npix(nside=nside)

def gen_fg_cl():
    cls_fg = np.load('../data/debeam_full_b/cl_fg.npy')
    Cl_TT = cls_fg[0]
    Cl_EE = cls_fg[1]
    Cl_BB = cls_fg[2]
    Cl_TE = np.zeros_like(Cl_TT)
    return np.array([Cl_TT, Cl_EE, Cl_BB, Cl_TE])

def gen_map(rlz_idx=0, mode='mean', return_noise=False):
    # mode can be mean or std
    noise_seed = np.load('../../seeds_noise_2k.npy')
    cmb_seed = np.load('../../seeds_cmb_2k.npy')
    nside = 2048

    nstd = np.load(f'../../../FGSim/NSTDNORTH/2048/{freq}.npy')
    npix = hp.nside2npix(nside=2048)
    np.random.seed(seed=noise_seed[rlz_idx])
    # noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))
    noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))
    print(f"{np.std(noise[1])=}")

    if return_noise:
        return noise

    ps = np.load(f'../../../fitdata/2048/PS/{freq}/ps.npy')
    fg = np.load(f'../../../fitdata/2048/FG/{freq}/fg.npy')

    cls = np.load('../../../src/cmbsim/cmbdata/cmbcl_8k.npy')
    if mode=='std':
        np.random.seed(seed=cmb_seed[rlz_idx])
    elif mode=='mean':
        np.random.seed(seed=cmb_seed[0])

    cmb_iqu = hp.synfast(cls.T, nside=nside, fwhm=np.deg2rad(beam)/60, new=True, lmax=3*nside-1)

    pcfn = noise + ps + cmb_iqu + fg
    cfn = noise + cmb_iqu + fg
    n = noise
    return pcfn, noise, cfn

m_pcfn, m_n, m_cfn = gen_map(rlz_idx=rlz_idx, mode='std')
# m_b = hp.alm2map(hp.map2alm(m)[2], nside=2048)

mask = hp.read_map(f'./new_mask/mask_only_edge.fits')

obj = EBLeakageCorrection(m_pcfn, lmax=lmax, nside=nside, mask=mask, post_mask=mask)
_,_,cln_b_pcfn = obj.run_eblc()
slope_in = obj.return_slope()

# path_input = Path('./input_std_new')
# path_input.mkdir(exist_ok=True, parents=True)
# hp.write_map(path_input / Path(f'{rlz_idx}.fits'), cln_b_pcfn, overwrite=True)

# obj = EBLeakageCorrection(m_cfn, lmax=lmax, nside=nside, mask=mask, post_mask=mask, slope_in=slope_in)
# _,_,cln_b_cfn = obj.run_eblc()

# path_input = Path('./input_cfn_new')
# path_input.mkdir(exist_ok=True, parents=True)
# hp.write_map(path_input / Path(f'{rlz_idx}.fits'), cln_b_cfn, overwrite=True)

obj = EBLeakageCorrection(m_n, lmax=lmax, nside=nside, mask=mask, post_mask=mask, slope_in=slope_in)
_,_,cln_b_n = obj.run_eblc()

path_input = Path('./input_n_new')
path_input.mkdir(exist_ok=True, parents=True)
hp.write_map(path_input / Path(f'{rlz_idx}.fits'), cln_b_n, overwrite=True)










