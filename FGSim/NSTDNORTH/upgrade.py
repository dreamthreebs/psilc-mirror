import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

m = np.load('./30.npy')[1]

m2048 = hp.ud_grade(m, nside_out=2048, power=1)

hp.mollview(m2048)
plt.show()
