import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

lmax=2000

bin_qucleaned_B = np.load('./fitQU/cleaned_B.npy')
bin_qucorrupted_B = np.load('./fitQU/corrupted_B.npy')

bin_bcleaned_B = np.load('./fitB/cleaned_B.npy')
bin_bcorrupted_B = np.load('./fitB/corrupted_B.npy')

cl_bin_qu_clean = hp.anafast(bin_qucleaned_B, lmax=lmax)
cl_bin_qu_corrupt = hp.anafast(bin_qucorrupted_B, lmax=lmax)

cl_bin_b_clean = hp.anafast(bin_bcleaned_B, lmax=lmax)
cl_bin_b_corrupt = hp.anafast(bin_bcorrupted_B, lmax=lmax)

l = np.arange(lmax+1)

plt.loglog(l*(l+1)*cl_bin_b_clean/(2*np.pi), label='b corrected')
plt.loglog(l*(l+1)*cl_bin_b_corrupt/(2*np.pi), label='b corrupted')
plt.loglog(l*(l+1)*cl_bin_qu_clean/(2*np.pi), label='qu corrected')
plt.loglog(l*(l+1)*cl_bin_qu_corrupt/(2*np.pi), label='qu corrupted')

plt.legend()
plt.show()
