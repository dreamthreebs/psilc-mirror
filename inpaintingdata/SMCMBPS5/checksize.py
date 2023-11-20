import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

m_true9 = np.load('../CMB5/270.npy')[0]
m_true17 = np.load('../CMB5/155.npy')[0]

lmax = 1700
bl30 = hp.gauss_beam(fwhm=np.deg2rad(30)/60, lmax=lmax)
bl17 = hp.gauss_beam(fwhm=np.deg2rad(17)/60, lmax=lmax)
bl9 = hp.gauss_beam(fwhm=np.deg2rad(9)/60, lmax=lmax)



fold_list = [1.0, 1.25, 1.5, 1.75, 2.0]
l = np.arange(lmax+1)

for fold in fold_list:
    m9 = hp.read_map(f'./9arcmin/INPAINT/{fold}/95.fits', field=0)
    m17 = hp.read_map(f'./17arcmin/INPAINT/{fold}/95.fits', field=0)

    m_diff_9 = m9 - m_true9
    m_diff_17 = m17 - m_true17

    cl_diff9 = hp.anafast(m_diff_9, lmax=lmax)
    cl_diff17 = hp.anafast(m_diff_17, lmax=lmax)

    # plt.semilogy(l*(l+1)*cl_diff9/(2*np.pi)/bl9**2, label=f'cl diff 9 {fold}*beam size')
    plt.semilogy(l*(l+1)*cl_diff17/(2*np.pi)/bl17**2, label=f'cl diff 17 {fold}*beam size')

cl_true9 = hp.anafast(m_true9, lmax=lmax)
plt.semilogy(l*(l+1)*cl_true9/(2*np.pi)/bl9**2, label='cl true', color='black')


plt.legend()
plt.ylim(1e-3, 1e6)
plt.xlabel('$\\ell$')
plt.ylabel('$D_\\ell$')
plt.show()

