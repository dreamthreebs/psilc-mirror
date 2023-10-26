import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

m1 = np.load('./CMB/B/data.npy')[0]
m2 = np.load('./CMB/B/data.npy')[7]

bin_mask = np.load('../../../mask/north/BINMASKG.npy')
apo_mask = np.load('../../../mask/north/APOMASKC1_5.npy')

hp.orthview(m1*apo_mask, rot=[100,50,0], half_sky=True, min=-0.8, max=0.8, title='30')
hp.orthview(m2*apo_mask, rot=[100,50,0], half_sky=True, min=-0.8, max=0.8, title='270')
plt.show()


