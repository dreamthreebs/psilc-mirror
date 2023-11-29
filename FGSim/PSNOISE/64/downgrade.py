import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from pathlib import Path
import glob

nside = 64
m = np.load('../40.npy')

m64 = hp.ud_grade(m, nside_out=nside)
# np.save('./40.npy', m64)

hp.mollview(m64[0], norm='hist')
plt.show()

