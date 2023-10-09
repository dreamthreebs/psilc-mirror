import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

cmbcl = np.load('./cmbdata/cmbcl3.npy') # (n_ell, n_cl) TT, EE, BB, PP
l = np.arange(cmbcl.shape[0])
print(f'{cmbcl.shape}')

lmax = 2000
nside = 2048

# for index, cmb_type in enumerate('TEB'):
#     cl = cmbcl[:,index]
#     m = hp.synfast(cl, nside, lmax)
#     hp.mollview(m, norm='hist')
#     plt.show()
#     np.save(f'./cmbdata/{cmb_type}/{cmb_type}cmbmap.npy', m)

plt.plot(l*(l+1)*cmbcl[:,3]/(2*np.pi));plt.show()
m = hp.synfast(cmbcl.T, nside=nside, lmax=lmax, new=True)
print(f"{m.shape}")

cl = hp.anafast(m, lmax=lmax)
l = np.arange(cl.shape[1])

for i in range(3):
    plt.loglog(l*(l+1)*cl[i]/(2*np.pi))
    plt.show()

# np.save('./cmbdata/cmbtqunoB20483.npy', m)


