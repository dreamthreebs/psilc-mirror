import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from cmbutils.map import calc_fsky

ps_mask = np.load("./inpainting/new_mask/apo_ps_mask.npy")
apo_mask = np.load('../../psfit/fitv4/fit_res/2048/ps_mask/new_mask/apo_C1_3_apo_3_apo_3.npy')

hp.orthview(ps_mask, rot=[100,50,0], title='ps_mask')
hp.orthview(apo_mask, rot=[100,50,0], title='apo')
plt.show()

fsky_ps_mask = calc_fsky(ps_mask)
fsky_apo_mask = calc_fsky(apo_mask)
print(f"{fsky_ps_mask=}, {fsky_apo_mask=}")
