import numpy as np
import matplotlib.pyplot as plt
import healpy as hp


cmb = np.load('../../../FGSim/CMB/270.npy')

cl_cmb = hp.anafast(cmb, lmax=300)

cl = np.load('./nilc_cl1.npy')
print(cl.shape)
l = np.arange(len(cl))
plt.plot(l*(l+1)*cl_cmb[1]/(2*np.pi))
plt.plot(l*(l+1)*cl/(2*np.pi));plt.show()

