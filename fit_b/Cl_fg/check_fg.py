import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

cl_fg = np.load('./data_debeam/cl_fg.npy')
print(f'{cl_fg.shape=}')
l = np.arange(np.size(cl_fg, axis=1))
plt.plot(l*(l+1)*cl_fg[2]/(2*np.pi), label='debeam')

cl_fg_b = np.load('./data_old/cl_fg_BB.npy')
l = np.arange(np.size(cl_fg_b))
plt.plot(l*(l+1)*cl_fg_b/(2*np.pi), label='no debeam')
plt.loglog()
plt.legend()
plt.xlabel("$\\ell$")
plt.ylabel("$D_\\ell^{BB} [\mu K^2]$")
plt.show()



