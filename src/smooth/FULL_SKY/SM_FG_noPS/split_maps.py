import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import glob
from pathlib import Path

sim_pos = glob.glob('./*.npy')
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

np.save(f'./T/fg.npy', T)
np.save(f'./E/fg.npy', E)
np.save(f'./B/fg.npy', B)


# for file in sim_pos:
#     freq = int(Path(file).stem)
#     print(f'{freq = }')
#     m = np.load(file)
#     for index, m_type in enumerate('TEB'):
#         # hp.mollview(m[index],norm='hist');plt.show()
#         np.save(f'./{m_type}/{freq}.npy', m[index])


