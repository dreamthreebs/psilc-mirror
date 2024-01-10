import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

cmbcl = np.load('./cmbdata/cmbcl.npy') # (n_ell, n_cl) TT, EE, BB, TE
l = np.arange(cmbcl.shape[0])
print(f'{cmbcl.shape}')

lmax = 500
nside = 2048

# for i in range(50):
# print(f'{i=}')
m = hp.synfast(cmbcl.T, nside=nside, lmax=lmax, new=True)

# for index, cmb_type in enumerate('TQU'):
#     hp.mollview(m[index], norm='hist')
#     plt.show()

np.save(f'./cmbdata/lmax500.npy', m)
# hp.write_map(f'./cmbdata/1.fits', m)


