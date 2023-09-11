import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

cl1 = np.load('./nilc_noise_cl20.npy')

cl2 = np.load('./nilc_noise_cl2avg.npy')
l = np.arange(len(cl1))

plt.plot(l*(l+1)*cl1/(2*np.pi))
plt.plot(l*(l+1)*cl2/(2*np.pi))
plt.show()
