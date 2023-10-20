import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

data = np.load('./data.npy')
for i in range(8):
    hp.orthview(data[i], rot=[100,50,0], half_sky=True)
    plt.show()

