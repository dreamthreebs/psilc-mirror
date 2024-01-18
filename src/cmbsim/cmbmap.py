import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import os

cmbcl = np.load('./cmbdata/cmbcl.npy') # (n_ell, n_cl) TT, EE, BB, TE
l = np.arange(cmbcl.shape[0])
print(f'{cmbcl.shape}')

lmax = 1000
nside = 256

if not os.path.exists('./cmbdata/rlz_nside256'):
    os.mkdir('./cmbdata/rlz_nside256')
for i in range(200):
    print(f'{i=}')
    m = hp.synfast(cmbcl.T[0], nside=nside, lmax=lmax, new=True)

# for index, cmb_type in enumerate('TQU'):
#     hp.mollview(m[index], norm='hist')
#     plt.show()


    np.save(f'./cmbdata/rlz_nside256/{i}.npy', m)
# hp.write_map(f'./cmbdata/1.fits', m)


