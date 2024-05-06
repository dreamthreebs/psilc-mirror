import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

cl = np.load('./cmbcl.npy')
print(f'{cl.shape=}')

lmax = 1999
l = np.arange(lmax+1)
dl_factor = l*(l+1)/(2*np.pi)

for i in range(cl.shape[-1]):
    print(f'{i=}')
    plt.loglog(np.abs(cl[:lmax+1,i]*dl_factor), label=f'{i}')

plt.legend()
plt.show()



