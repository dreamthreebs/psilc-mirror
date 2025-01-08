import numpy as np
import healpy as hp
import matplotlib.pyplot as plt


apo_ps_mask = np.load('./mask/apo_ps_mask.npy')
mask = hp.read_map('./mask/mask_add_edge.fits')
fsky = np.sum(apo_ps_mask) / np.size(apo_ps_mask)
print(f'{fsky=}')

hp.orthview(mask, rot=[100,50,0], title='mask')
hp.orthview(apo_ps_mask, rot=[100,50,0], title='apo_ps_mask')
plt.show()


input_mean = hp.read_map('./input_mean/1.fits')
# input_std = hp.read_map('./input_std/1.fits')
input_n = hp.read_map('./input_n/1.fits')

output_mean = hp.read_map('./output_m2_mean/1.fits')
# output_std = hp.read_map('./output_m2_std/1.fits')
output_n = hp.read_map('./output_m2_n/1.fits')

hp.orthview(input_mean, rot=[100,50,0], title='input mean')
# hp.orthview(input_std, rot=[100,50,0], title='input std')
hp.orthview(input_n, rot=[100,50,0], title='input n')

hp.orthview(output_mean, rot=[100,50,0], title='output mean')
# hp.orthview(output_std, rot=[100,50,0], title='output std')
hp.orthview(output_n, rot=[100,50,0], title='output n')

plt.show()




