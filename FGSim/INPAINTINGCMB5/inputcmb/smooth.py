import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import glob

fits_file = sorted(glob.glob('../../../FG5/strongps/*.npy'), key=lambda x:int(Path(x).stem))
print(f"{fits_file=}")
cmb_file = sorted(glob.glob('../../CMB5/*.npy'), key=lambda x:int(Path(x).stem))
print(f"{cmb_file=}")

df = pd.read_csv('../../FreqBand5')
lmax = 350
nside = 512
directory = Path('./smoothcmb')
directory.mkdir(parents=True, exist_ok=True)

bl_std = hp.gauss_beam(fwhm=np.deg2rad(11)/60, lmax=lmax)

for i, (ps_pos, cmb_pos) in enumerate(zip(fits_file, cmb_file)):
    print(f'{ps_pos=}')
    ps = np.load(ps_pos)[0]
    cmb = np.load(cmb_pos)[0]
    m = cmb
    hp.mollview(m, norm='hist')
    plt.show()
    freq = df.at[i, 'freq']
    beam = df.at[i, 'beam']
    print(f'{freq=}, {beam=}')

    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax)
    debeamed_m = hp.alm2map(hp.almxfl(hp.map2alm(m, lmax=lmax), bl_std/bl), nside=nside)
    hp.write_map(f'./smoothcmb/{freq}.fits', debeamed_m, overwrite=True)


#     hp.mollview(m)
#     plt.show()
