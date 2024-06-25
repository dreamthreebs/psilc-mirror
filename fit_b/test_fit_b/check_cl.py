import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

m = np.load('../../fitdata/2048/CMB/270/1.npy')
cl = hp.anafast(m)

l = np.arange(cl.shape[1])
plt.plot(l, l*(l+1)*cl[0]/(2*np.pi))
plt.plot(l, l*(l+1)*cl[1]/(2*np.pi))
plt.plot(l, l*(l+1)*cl[2]/(2*np.pi))
plt.loglog()
plt.show()

