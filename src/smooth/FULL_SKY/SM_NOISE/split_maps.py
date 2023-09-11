import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import glob
from pathlib import Path

for i in range(15,30):

    sim_pos = glob.glob(f'./{i}/*.npy')
    Tlist = []
    Elist = []
    Blist = []
    
    for file in sim_pos:
        freq = int(Path(file).stem)
        print(f'{freq = }')
        m = np.load(file)
        Tlist.append(m[0])
        Elist.append(m[1])
        Blist.append(m[2])
    
    T = np.array(Tlist)
    E = np.array(Elist)
    B = np.array(Blist)
    
    np.save(f'./{i}/T/noise.npy', T)
    np.save(f'./{i}/E/noise.npy', E)
    np.save(f'./{i}/B/noise.npy', B)


# for file in sim_pos:
#     freq = int(Path(file).stem)
#     print(f'{freq = }')
#     m = np.load(file)
#     for index, m_type in enumerate('TEB'):
#         # hp.mollview(m[index],norm='hist');plt.show()
#         np.save(f'./{m_type}/{freq}.npy', m[index])


