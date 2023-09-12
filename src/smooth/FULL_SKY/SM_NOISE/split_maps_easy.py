import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import glob
from pathlib import Path

sim_pos = glob.glob(f'./*.npy')
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

np.save(f'./T/noise.npy', T)
np.save(f'./E/noise.npy', E)
np.save(f'./B/noise.npy', B)


