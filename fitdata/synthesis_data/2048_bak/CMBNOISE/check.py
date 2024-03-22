import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

m = np.load('./155/0.npy')
m1 = np.load('./155/55.npy')

hp.mollview(m[0])
hp.mollview(m[1])
plt.show()

