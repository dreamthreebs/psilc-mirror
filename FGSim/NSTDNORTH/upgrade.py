import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

freqlist = [30, 40, 85, 95, 145, 155, 215, 270 ]
nside_out = 512

for freq in freqlist:

    m = np.load(f'./{freq}.npy')
    
    ud_m = hp.ud_grade(m, nside_out=nside_out, power=1)
    
    # hp.mollview(m2048)
    # plt.show()
    
    np.save(f'./{nside_out}/{freq}.npy', ud_m)
