import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

noise = np.load('./145.npy')
for i in range(3):
    hp.mollview(noise[i])
    plt.show()
