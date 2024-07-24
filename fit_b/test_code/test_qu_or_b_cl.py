import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

nside = 1024
l = np.arange(3*nside)
dl_factor = l*(l+1)/(2*np.pi)
m_qu_b = np.load('./qu_or_b_noise/m_qu_b_1.npy')
m_b = np.load('./qu_or_b_noise/m_b_1.npy')

cl_qu_b = hp.anafast(m_qu_b)
cl_b = hp.anafast(m_b)
plt.loglog(cl_qu_b*dl_factor)
plt.loglog(cl_b*dl_factor)
plt.show()



