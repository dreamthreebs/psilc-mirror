import numpy as np
import healpy as hp
import matplotlib.pyplot as plt


apo_ps_mask = np.load('./new_mask/apo_ps_mask.npy')
mask = hp.read_map('./new_mask/mask_only_edge.fits')
mask_2d5 = hp.read_map('./new_mask/mask2d5.fits')
fsky = np.sum(apo_ps_mask) / np.size(apo_ps_mask)
print(f'{fsky=}')

hp.orthview(mask, rot=[100,50,0], title='mask')
hp.orthview(apo_ps_mask, rot=[100,50,0], title='apo_ps_mask')
hp.orthview(mask_2d5, rot=[100,50,0], title='mask2d5')
plt.show()


input_std = hp.read_map('./input_std_new/1.fits')
input_cfn = hp.read_map('./input_cfn_new/1.fits')

output_std = hp.read_map('./output_m3_std_new/1.fits')
# output_n = hp.read_map('./output_m3_n_new/1.fits')

hp.orthview(input_std, rot=[100,50,0], title='input std', half_sky=True)
hp.orthview(input_cfn, rot=[100,50,0], title='input cfn', half_sky=True)
hp.orthview(output_std, rot=[100,50,0], title='output std', half_sky=True)
hp.orthview(input_std - input_cfn, rot=[100,50,0], title='residual pcfn-cfn', half_sky=True)
hp.orthview(input_std - output_std, rot=[100,50,0], title='residual pcfn-inp', half_sky=True)
hp.orthview(output_std - input_cfn, rot=[100,50,0], title='residual inp-cfn', half_sky=True)

# hp.orthview(output_std, rot=[100,50,0], title='output std', half_sky=True)
# hp.orthview(output_n, rot=[100,50,0], title='output n', half_sky=True)

plt.show()





