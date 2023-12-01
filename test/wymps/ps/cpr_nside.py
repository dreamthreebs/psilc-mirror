import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

m512 = np.load('./512.npy')
m2048_512 = np.load('./d_2048_512.npy')

hp.gnomview(m512, title='512')
hp.gnomview(m2048_512, title='2048 to 512')
plt.show()
