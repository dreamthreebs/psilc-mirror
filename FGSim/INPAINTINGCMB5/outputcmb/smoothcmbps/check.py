import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

m = hp.read_map('./215.fits', field=0)
hp.mollview(m)
plt.show()

