import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from pathlib import Path
from eblc_base_slope import EBLeakageCorrection

rlz_idx=0
nside = 2048
beam = 11
lmax = 2500
cmb_seed = np.load('../../seeds_cmb_2k.npy')

def gen_map():

    cls = np.load('../../../src/cmbsim/cmbdata/cmbcl_8k.npy')
    np.random.seed(seed=cmb_seed[rlz_idx])
    # cmb_iqu = hp.synfast(cls.T, nside=nside, fwhm=np.deg2rad(beam)/60, new=True, lmax=1999)
    cmb_iqu = hp.synfast(cls.T, nside=nside, fwhm=np.deg2rad(beam)/60, new=True, lmax=lmax)

    m = cmb_iqu
    return m

m = gen_map()
mask = hp.read_map(f'./mask/mask.fits')
obj = EBLeakageCorrection(m, lmax=lmax, nside=nside, mask=mask, post_mask=mask)
_,_,cln_b = obj.run_eblc()
slope = obj.return_slope()
print(f'{slope=}')

path_slope = Path(f'./eblc_slope')
path_slope.mkdir(exist_ok=True, parents=True)

np.save(path_slope / Path(f'{rlz_idx}.npy'), slope)








