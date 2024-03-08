import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

m = hp.read_map('./1.fits')
hp.orthview(m, rot=[100,50,0], half_sky=True)
plt.show()
