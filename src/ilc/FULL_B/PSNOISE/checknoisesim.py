import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

cl1 = np.load('./nilc_noise_cl8sim0.npy')

cl2 = np.load('./nilc_noise_cl8avg.npy')
l = np.arange(len(cl1))

plt.plot(l*(l+1)*cl1/(2*np.pi))
plt.plot(l*(l+1)*cl2/(2*np.pi))
plt.show()
