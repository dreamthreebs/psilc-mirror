import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

lmax=2000
freq = 270
l = np.arange(lmax+1)

m = hp.read_map(f'./{freq}.fits', field=0)

m_cmb = np.load(f'../../../../inpaintingdata/CMB5/{freq}.npy')[0]
m_cmbps = hp.read_map(f'../../../../inpaintingdata/CMBPS5/{freq}.fits', field=0)

hp.mollview(m, norm='hist', title='inpainted map')
plt.show()
cl_inpaint = hp.anafast(m, lmax=lmax)
cl_cmb = hp.anafast(m_cmb, lmax=lmax)
cl_cmbps = hp.anafast(m_cmbps, lmax=lmax)
bl = hp.gauss_beam(fwhm=np.deg2rad(9)/60, lmax=lmax)

plt.semilogy(l*(l+1)*cl_cmb/(2*np.pi)/bl**2, label='cmb only')
plt.semilogy(l*(l+1)*cl_inpaint/(2*np.pi)/bl**2, label='inpainted')
plt.semilogy(l*(l+1)*cl_cmbps/(2*np.pi)/bl**2, label='cmb + ps')
plt.xlabel('$\\ell$')
plt.ylabel('$D_\\ell$')
plt.ylim(1e-5, 1e8)

plt.legend()
plt.show()
