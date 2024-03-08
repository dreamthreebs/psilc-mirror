import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

mask = np.load('../../../../../../../src/mask/north/BINMASKG2048.npy')
m = hp.read_map('./0.fits')

hp.orthview(m*mask, rot=[100,50,0],half_sky=True)
plt.show()


