import numpy as np
import healpy as hp
import matplotlib.pyplot as plt


apo_ps_mask = np.load('./new_mask/apo_ps_mask.npy')
apo_2_mask = np.load(f'./new_mask/apo_ps_mask_2degree.npy')
apo_3_mask = np.load(f'./new_mask/apo_ps_mask_3degree.npy')
apo_4_mask = np.load(f'./new_mask/apo_ps_mask_4degree.npy')
# mask = hp.read_map('./new_mask/mask_only_edge.fits')
# mask_2d5 = hp.read_map('./new_mask/mask2d5.fits')
fsky = np.sum(apo_ps_mask) / np.size(apo_ps_mask)
print(f'{fsky=}')

# hp.orthview(mask, rot=[100,50,0], title='mask')
hp.orthview(apo_ps_mask, rot=[100,50,0], title='apo_ps_mask')
hp.orthview(apo_2_mask, rot=[100,50,0], title='apo_ps_mask 2 degree')
hp.orthview(apo_3_mask, rot=[100,50,0], title='apo_ps_mask 3 degree')
hp.orthview(apo_4_mask, rot=[100,50,0], title='apo_ps_mask 4 degree')
# hp.orthview(mask_2d5, rot=[100,50,0], title='mask2d5')
plt.show()


input_std = hp.read_map('./input_std_new/10.fits')
input_n = hp.read_map('./input_n_new/10.fits')

output_std = hp.read_map('./output_m4_std_new/10.fits')
output_n = hp.read_map('./output_m4_n_new/10.fits')

hp.orthview(input_std, rot=[100,50,0], title='input std', half_sky=True)
hp.orthview(input_n, rot=[100,50,0], title='input n', half_sky=True)

hp.orthview(output_std, rot=[100,50,0], title='output std', half_sky=True)
hp.orthview(output_n, rot=[100,50,0], title='output n', half_sky=True)

plt.show()





