import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

freq = 30
lmax = 1999
beam = 63
nside = 2048
bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax)
l = np.arange(lmax+1)
# m = np.load('./pcn_after_removal/2sigma/B/map_cln_b1.npy')
m_origin = np.load('../../../../src/cmbsim/cmbdata/m_realization/1.npy')
B_ori = hp.alm2map(hp.map2alm(m_origin, lmax=lmax)[2], nside=nside)
m = np.load(f'../../../../fitdata/2048/CMB/{freq}/1.npy')
B = hp.alm2map(hp.map2alm(m, lmax=lmax)[2], nside=nside)

# apo_mask = np.load('../../../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5APO_5.npy')

cl = hp.anafast(B, lmax=lmax)
cl_ori = hp.anafast(B_ori, lmax=lmax)
plt.plot(l*(l+1)*cl/(2*np.pi)/bl**2)
plt.plot(l*(l+1)*cl_ori/(2*np.pi))
plt.semilogy()
plt.show()

