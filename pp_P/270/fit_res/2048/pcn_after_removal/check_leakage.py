import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from eblc_base import EBLeakageCorrection

lmax = 1999
l = np.arange(lmax+1)
nside = 2048

m_cmb = np.load('../../../../../fitdata/2048/CMB/270/1.npy')
bin_mask = np.load('../../../../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5.npy')
apo_mask = np.load('../../../../../psfit/fitv4/fit_res/2048/ps_mask/no_edge_mask/C1_5APO_5.npy')
# m_removal_b = np.load('./2sigma/B/map_cln_b1.npy')

full_E = hp.alm2map(hp.map2alm(m_cmb, lmax=lmax)[1], nside=nside)
full_B = hp.alm2map(hp.map2alm(m_cmb, lmax=lmax)[2], nside=nside)

cut_E = hp.alm2map(hp.map2alm(m_cmb*bin_mask, lmax=lmax)[1], nside=nside)
cut_B = hp.alm2map(hp.map2alm(m_cmb*bin_mask, lmax=lmax)[2], nside=nside)

cl_full_E = hp.anafast(full_E * bin_mask, lmax=lmax)
cl_cut_E = hp.anafast(cut_E * bin_mask, lmax=lmax)

cl_full_B = hp.anafast(full_B * bin_mask, lmax=lmax)
cl_cut_B = hp.anafast(cut_B * bin_mask, lmax=lmax)

obj = EBLeakageCorrection(m=m_cmb*bin_mask, lmax=lmax, nside=nside, mask=bin_mask, post_mask=bin_mask)
_,_,cln_B = obj.run_eblc()
cl_eblc_B = hp.anafast(cln_B, lmax=lmax)

np.save('./test_leakage/cl_full_E.npy', cl_full_E)
np.save('./test_leakage/cl_full_B.npy', cl_full_B)
np.save('./test_leakage/cl_cut_E.npy', cl_full_E)
np.save('./test_leakage/cl_cut_B.npy', cl_full_B)
np.save('./test_leakage/cl_eblc_B.npy', cl_eblc_B)

dl_factor = l*(l+1) / (2*np.pi)
plt.plot(l, cl_full_E * dl_factor, label='full E')
plt.plot(l, cl_cut_E * dl_factor, label='cut E')

plt.plot(l, cl_full_B * dl_factor, label='full B')
plt.plot(l, cl_cut_B * dl_factor, label='cut B')
plt.plot(l, cl_eblc_B * dl_factor, label='EBLeakageCorrection B')
plt.semilogy()
plt.legend()
plt.xlabel('$\\ell$')
plt.ylabel('$D_\\ell [\mu K^2]$')
plt.show()




# hp.orthview(full_E*bin_mask, rot=[100,50,0], title='full E', half_sky=True)
# hp.orthview(cut_E*bin_mask, rot=[100,50,0], title='cut E', half_sky=True)
# hp.orthview((full_E-cut_E)*bin_mask, rot=[100,50,0], title='full - cut E', half_sky=True)

# hp.orthview(full_B*bin_mask, rot=[100,50,0], title='full B', half_sky=True)
# hp.orthview(cut_B*bin_mask, rot=[100,50,0], title='cut B', half_sky=True)
# hp.orthview((full_B-cut_B)*bin_mask, rot=[100,50,0], title='full - cut B', half_sky=True)
# plt.show()




