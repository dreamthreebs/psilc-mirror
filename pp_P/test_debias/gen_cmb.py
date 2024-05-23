import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from pathlib import Path

lmax = 400
nside = 256
beam = 63


cl = np.load('../../src/cmbsim/cmbdata/cmbcl.npy')
print(f'{cl.shape=}')
path_cmb = Path('./cmb')
path_cmb.mkdir(exist_ok=True, parents=True)

for rlz_idx in range(1000):
    print(f'{rlz_idx=}')
    m = hp.synfast(cl.T, nside=nside, lmax=lmax, fwhm=np.deg2rad(beam)/60, new=True)
    np.save(path_cmb / Path(f'{rlz_idx}.npy'), m)

# cl_out = hp.anafast(m, lmax=4*nside)
# plt.semilogy(cl_out[2])
# plt.show()

