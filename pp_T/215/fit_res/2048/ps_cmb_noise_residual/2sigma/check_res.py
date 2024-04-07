import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

lmax = 2000
l = np.arange(lmax+1)
bl = hp.gauss_beam(fwhm=np.deg2rad(11)/60, lmax=lmax)
mask = np.load('../../../../../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5APO_5.npy')
pscmbnoise = np.load('../../../../../../fitdata/synthesis_data/2048/PSCMBNOISE/215/0.npy')[0]
cmbnoise = np.load('../../../../../../fitdata/synthesis_data/2048/CMBNOISE/215/0.npy')[0]

removal = np.load('./map0.npy') + cmbnoise


cl_pscmbnoise = hp.anafast(pscmbnoise * mask, lmax=lmax)
cl_cmbnoise = hp.anafast(cmbnoise * mask, lmax=lmax)
cl_removal = hp.anafast(removal * mask, lmax=lmax)
plt.plot(l*(l+1)*cl_pscmbnoise/(2*np.pi)/bl**2, label='pscmbnoise')
plt.plot(l*(l+1)*cl_cmbnoise/(2*np.pi)/bl**2, label='pscmbnoise')
plt.plot(l*(l+1)*cl_removal/(2*np.pi)/bl**2, label='removal')

plt.legend()
plt.show()
