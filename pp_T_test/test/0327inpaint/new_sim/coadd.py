import numpy as np
import healpy as hp
import matplotlib.pyplot as plt


ps_sim = np.load('./ps_sim_17.npy')
for i in range(10,40):
    print(f'{i=}')
    cmb_noise = np.load(f'../../../../fitdata/synthesis_data/2048/CMBNOISE/155/{i}.npy')[0]
    sim = cmb_noise +  ps_sim
    
    hp.write_map(f'./input/{i}.fits', m=sim, overwrite=True)
    
    # hp.mollview(sim, title='ps cmb noise')
    # plt.show()
