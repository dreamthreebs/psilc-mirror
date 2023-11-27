import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

freq = 270
lmax = 2000
l = np.arange(lmax+1)

bl = hp.gauss_beam(fwhm=np.deg2rad(9)/60, lmax=lmax)

cmbps = hp.read_map(f'../../inpaintingdata/CMBPS5/{freq}.fits', field=0)
cmb = np.load(f'../../inpaintingdata/CMB5/{freq}.npy')[0]

mask1 = hp.read_map(f'./i_mask1/1.0/{freq}.fits', field=0)
mask10 = hp.read_map(f'./i_mask10/1.0/{freq}.fits', field=0)
mask20 = hp.read_map(f'./i_mask20/1.0/{freq}.fits', field=0)
maskdot1 = hp.read_map(f'./i_maskdot1/1.0/{freq}.fits', field=0)
maskdot5 = hp.read_map(f'./i_maskdot5/1.0/{freq}.fits', field=0)

def calc_fsky(mask):
    return np.sum(mask)/np.size(mask)

fsky1 = calc_fsky(mask1)
fsky10 = calc_fsky(mask10)
fsky20 = calc_fsky(mask20)
fskydot1 = calc_fsky(maskdot1)
fskydot5 = calc_fsky(maskdot5)


cl_cmb = hp.anafast(cmb, lmax=lmax)

cl_cmb1 = hp.anafast(cmb * mask1, lmax=lmax) / fsky1
cl_cmb10 = hp.anafast(cmb * mask10, lmax=lmax) / fsky10
cl_cmb20 = hp.anafast(cmb * mask20, lmax=lmax) / fsky20
cl_cmbdot1 = hp.anafast(cmb * maskdot1, lmax=lmax) / fskydot1
cl_cmbdot5 = hp.anafast(cmb * maskdot5, lmax=lmax) / fskydot5

cl_cmbps = hp.anafast(cmbps, lmax=lmax)

# cl_cmbps1 = hp.anafast(cmbps * mask1, lmax=lmax) / fsky1
# cl_cmbps10 = hp.anafast(cmbps * mask10, lmax=lmax) / fsky10
# cl_cmbps20 = hp.anafast(cmbps * mask20, lmax=lmax) / fsky20
# cl_cmbpsdot1 = hp.anafast(cmbps * maskdot1, lmax=lmax) / fskydot1
# cl_cmbpsdot5 = hp.anafast(cmbps * maskdot5, lmax=lmax) / fskydot5

plt.semilogy(l*(l+1)*cl_cmb/(2*np.pi)/bl**2, label='cmb')

plt.semilogy(l*(l+1)*cl_cmb1/(2*np.pi)/bl**2, label='cmb 1')
plt.semilogy(l*(l+1)*cl_cmb10/(2*np.pi)/bl**2, label='cmb 10')
plt.semilogy(l*(l+1)*cl_cmb20/(2*np.pi)/bl**2, label='cmb 20')
plt.semilogy(l*(l+1)*cl_cmbdot1/(2*np.pi)/bl**2, label='cmb dot1')
plt.semilogy(l*(l+1)*cl_cmbdot5/(2*np.pi)/bl**2, label='cmb dot5')

plt.semilogy(l*(l+1)*cl_cmbps/(2*np.pi)/bl**2, label='cmbps')

# plt.semilogy(l*(l+1)*cl_cmbps1/(2*np.pi)/bl**2, label='cmbps 1')
# plt.semilogy(l*(l+1)*cl_cmbps10/(2*np.pi)/bl**2, label='cmbps 10')
# plt.semilogy(l*(l+1)*cl_cmbps20/(2*np.pi)/bl**2, label='cmbps 20')
# plt.semilogy(l*(l+1)*cl_cmbpsdot1/(2*np.pi)/bl**2, label='cmbps dot1')
# plt.semilogy(l*(l+1)*cl_cmbpsdot5/(2*np.pi)/bl**2, label='cmbps dot5')

plt.legend()
plt.ylim(1e2, 1e6)
plt.xlabel('$\\ell$')
plt.ylabel('$D_\\ell$')
plt.show()








