import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

cmb = np.load('../../src/cmbsim/cmbdata/cmbmap.npy')[0]
mask_north = np.load('../../src/mask/north/BINMASKG2048.npy')
cmb[mask_north==0] = hp.UNSEEN

hp.mollview(cmb)
plt.show()

mask_arr = hp.ma(cmb, badval=hp.UNSEEN)
print(f'{mask_arr=}')

lmax = 500
cmb = np.load('../../src/cmbsim/cmbdata/cmbmap.npy')[0]
cl1= hp.anafast(cmb, lmax=lmax)
cl2 = hp.anafast(mask_arr, lmax=lmax)
cl3 = hp.anafast(cmb * mask_north, lmax=lmax)


plt.plot(cl1, linestyle='--')
plt.plot(cl2)
plt.plot(cl3)
plt.show()


''' Conclusion:
    the masked array is not used in the powerspectrum calculations
'''
