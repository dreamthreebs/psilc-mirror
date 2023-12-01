import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

freqlist = [30, 85, 95, 145, 155, 215, 270 ]

for freq in freqlist:

    m = np.load(f'./{freq}.npy')
    
    m2048 = hp.ud_grade(m, nside_out=2048, power=1)
    
    # hp.mollview(m2048)
    # plt.show()
    
    np.save(f'./2048/{freq}.npy', m2048)
