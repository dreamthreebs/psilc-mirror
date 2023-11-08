import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import glob
from pathlib import Path

fg = glob.glob('./*.npy')
lmax=300
nside=512

for file in fg:
    fg = np.load(file)
    freq = Path(file).stem
    print(f'{freq = }')
    fgalm = hp.map2alm(fg, lmax=lmax)
    T_fg = hp.alm2map(fgalm[0], nside=nside)
    E_fg = hp.alm2map(fgalm[1], nside=nside)
    B_fg = hp.alm2map(fgalm[2], nside=nside)
    hp.mollview(T_fg,norm='hist')
    hp.mollview(E_fg,norm='hist')
    hp.mollview(B_fg,norm='hist')
    plt.show()
