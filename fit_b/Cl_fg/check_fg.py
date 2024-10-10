import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

cl_fg = np.load('./data_1010/cl_fg.npy')
print(f'{cl_fg.shape=}')
l = np.arange(np.size(cl_fg, axis=1))
plt.plot(l*(l+1)*cl_fg[2]/(2*np.pi), label='estimate on debeam B(from full sky), add beam')
# plt.plot(l*(l+1)*cl_fg[1]/(2*np.pi), label='estimate on debeam E(from full sky), add beam')

cl_fg_b = np.load('./data_old/cl_fg_BB.npy')
# cl_fg_e = np.load('./data_old/cl_fg_EE.npy')

l = np.arange(np.size(cl_fg_b))
plt.plot(l*(l+1)*cl_fg_b/(2*np.pi), label='previous B')
# plt.plot(l*(l+1)*cl_fg_e/(2*np.pi), label='previous E')

cl_full_qu = np.load('./data_debeam_full_qu/cl_fg.npy')
l = np.arange(np.size(cl_full_qu, axis=1))
# plt.plot(l*(l+1)*cl_full_qu[2]/(2*np.pi), label='estimate from debeam QU(from full sky), add beam')
# plt.plot(l*(l+1)*cl_full_qu[1]/(2*np.pi), label='debeam qu E')

# cl_partial_b = np.load('./data_debeam_partial_b/cl_fg.npy')
# l = np.arange(np.size(cl_partial_b, axis=1))
# plt.plot(l*(l+1)*cl_partial_b[2]/(2*np.pi), label='estimate from debeam B(from partial sky), add beam')

# cl_partial_qu = np.load('./data_debeam_partial_qu/cl_fg.npy')
# l = np.arange(np.size(cl_partial_qu, axis=1))
# plt.plot(l*(l+1)*cl_partial_qu[2]/(2*np.pi), label='estimate from debeam QU(from partial sky), add beam')

cl_cmb = np.load('../../src/cmbsim/cmbdata/cmbcl_8k.npy').T[:,:np.size(cl_full_qu, axis=1)]
l = np.arange(np.size(cl_cmb, axis=1))
bl = hp.gauss_beam(fwhm=np.deg2rad(67)/60, lmax=np.size(cl_cmb, axis=1)-1)
plt.plot(l*(l+1)*cl_cmb[2]*bl**2/(2*np.pi), label='cmb, add beam')

plt.loglog()
plt.legend()
plt.xlabel("$\\ell$")
plt.ylabel("$D_\\ell^{BB} [\mu K^2]$")
plt.show()



