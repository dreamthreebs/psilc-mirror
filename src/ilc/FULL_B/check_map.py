import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

m = np.load('./nilc_fgres_map10.npy')

hp.mollview(m, norm='hist')
plt.show()

