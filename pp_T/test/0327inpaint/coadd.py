import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

cmb_noise = np.load('../../../fitdata/synthesis_data/2048/CMBNOISE/155/0.npy')[0]
ps_sim = np.load('./ps_sim.npy')
sim = cmb_noise +  ps_sim
np.save('cmb_noise.npy', cmb_noise)
np.save('ps_cmb_noise.npy', sim)

hp.mollview(ps_sim, title='ps')
hp.mollview(sim, title='ps cmb noise')
plt.show()
