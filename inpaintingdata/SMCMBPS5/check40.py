import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

m30 = hp.read_map('./30arcmin/INPAINT/1.0/40.fits', field=0)
m17 = hp.read_map('./17arcmin/INPAINT/1.0/40.fits', field=0)
m11 = hp.read_map('./11arcmin/INPAINT/1.0/40.fits', field=0)
m9 = hp.read_map('./9arcmin/INPAINT/1.0/40.fits', field=0)
m_true9 = np.load('../CMB5/270.npy')[0]
m_true17 = np.load('../CMB5/155.npy')[0]
m_true11 = np.load('../CMB5/215.npy')[0]

# m_cmbps_9 = hp.read_map('../CMBPS5/270.fits', field=0)
# m_cmbps_30 = hp.read_map('../CMBPS5/95.fits', field=0)
# m_cmbps_17 = hp.read_map('../CMBPS5/155.fits', field=0)

m_diff_9 = m9 - m_true9
m_diff_17 = m17 - m_true17
m_diff_11 = m11 - m_true11

# hp.mollview(m30, title='m30')
# hp.mollview(m17, title='m17')
# hp.mollview(m9, title='m9')

# hp.mollview(m_diff_9, norm='hist', title='m_diff_9')
# hp.mollview(m_diff_17, norm='hist', title='m_diff_17')
# hp.mollview(m_diff_11, norm='hist', title='m_diff_17')
# plt.show()


lmax = 750
l = np.arange(lmax+1)
bl30 = hp.gauss_beam(fwhm=np.deg2rad(30)/60, lmax=lmax)
bl17 = hp.gauss_beam(fwhm=np.deg2rad(17)/60, lmax=lmax)
bl11 = hp.gauss_beam(fwhm=np.deg2rad(11)/60, lmax=lmax)
bl9 = hp.gauss_beam(fwhm=np.deg2rad(9)/60, lmax=lmax)

cl_true9 = hp.anafast(m_true9, lmax=lmax)
cl30 = hp.anafast(m30, lmax=lmax)
cl17 = hp.anafast(m17, lmax=lmax)
cl11 = hp.anafast(m11, lmax=lmax)
cl9 = hp.anafast(m9, lmax=lmax)
cl_diff9 = hp.anafast(m_diff_9, lmax=lmax)
cl_diff17 = hp.anafast(m_diff_17, lmax=lmax)
cl_diff11 = hp.anafast(m_diff_11, lmax=lmax)

# cl_cmbps_9 = hp.anafast(m_cmbps_9, lmax=lmax)
# cl_cmbps_17 = hp.anafast(m_cmbps_17, lmax=lmax)
# cl_cmbps_30 = hp.anafast(m_cmbps_30, lmax=lmax)

plt.semilogy(l*(l+1)*cl_true9/(2*np.pi)/bl9**2, label='cl true')
plt.semilogy(l*(l+1)*cl30/(2*np.pi)/bl30**2, label='cl 30')
plt.semilogy(l*(l+1)*cl17/(2*np.pi)/bl17**2, label='cl 17')
plt.semilogy(l*(l+1)*cl11/(2*np.pi)/bl11**2, label='cl 11')
plt.semilogy(l*(l+1)*cl9/(2*np.pi)/bl9**2, label='cl 9')
plt.semilogy(l*(l+1)*cl_diff9/(2*np.pi)/bl9**2, label='cl diff 9')
plt.semilogy(l*(l+1)*cl_diff11/(2*np.pi)/bl11**2, label='cl diff 11')
plt.semilogy(l*(l+1)*cl_diff17/(2*np.pi)/bl17**2, label='cl diff 17')

# plt.semilogy(l*(l+1)*cl_cmbps_30/(2*np.pi)/bl30**2, label='cl cmbps 30')
# plt.semilogy(l*(l+1)*cl_cmbps_17/(2*np.pi)/bl17**2, label='cl cmbps 17')
# plt.semilogy(l*(l+1)*cl_cmbps_9/(2*np.pi)/bl9**2, label='cl cmbps 9')

plt.legend()
plt.ylim(1e-3, 1e6)
plt.xlabel('$\\ell$')
plt.ylabel('$D_\\ell$')
plt.show()

