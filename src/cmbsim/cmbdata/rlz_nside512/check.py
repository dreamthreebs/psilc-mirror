import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

m = np.load('./0.npy')
print(f'{m.shape=}')
hp.mollview(m)
plt.show()
