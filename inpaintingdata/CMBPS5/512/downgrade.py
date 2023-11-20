import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

lmax = 2000

m2048 = hp.read_map('../95.fits', field=0)

m512 = hp.ud_grade(m2048, nside_out=512)

cl2048 = hp.anafast(m2048, lmax=lmax)
cl512 = hp.anafast(m512, lmax=lmax)

plt.semilogy(cl2048, label='cl nside=2048')
plt.semilogy(cl512, label='cl nside=512')
plt.legend()
plt.show()

