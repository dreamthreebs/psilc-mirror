import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import os

cmbcl = np.load('./cmbdata/cmbcl.npy') # (n_ell, n_cl) TT, EE, BB, TE
l = np.arange(cmbcl.shape[0])
print(f'{cmbcl.shape}')

lmax = 2000
nside = 2048

if not os.path.exists('./cmbdata/m_realization'):
    os.mkdir('./cmbdata/m_realization')
for i in range(900,1000):
    print(f'{i=}')
    m = hp.synfast(cmbcl.T, nside=nside, lmax=lmax, new=True)

# for index, cmb_type in enumerate('TQU'):
#     hp.mollview(m[index], norm='hist')
#     plt.show()


    np.save(f'./cmbdata/m_realization/{i}.npy', m)
# hp.write_map(f'./cmbdata/1.fits', m)


