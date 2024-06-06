import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

lmax = 1500
nside = 512
# cl = np.load('../../../../../../src/cmbsim/cmbdata/cmbcl.npy')

# for rlz_idx in range(100):
#     print(f'{rlz_idx=}')
#     m = hp.synfast(cl.T, nside=nside, lmax=lmax, new=True)
#     np.save(f'./cmb/{rlz_idx}.npy', m)

nstd = np.load('../../../../../../FGSim/NSTDNORTH/270.npy')

# hp.mollview(nstd[0])
# hp.mollview(nstd[1])
# hp.mollview(nstd[2])
# plt.show()

for rlz_idx in range(100):
    print(f'{rlz_idx=}')
    noise = nstd * np.random.normal(loc=0, scale=1, size=(nstd.shape[0], nstd.shape[1]))
    np.save(f'./noise/{rlz_idx}.npy', noise)


