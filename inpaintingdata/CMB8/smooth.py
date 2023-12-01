import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt

cmb_ini = np.load('../../src/cmbsim/cmbdata/cmbmap.npy')
print(f'{cmb_ini.shape=}')

lmax = 2000
df = pd.read_csv('../../FGSim/FreqBand5')

for i in range(len(df)):
    freq = df.at[i, 'freq']
    beam = df.at[i, 'beam']
    print(f'{freq=}, {beam=}')
    cmb = hp.smoothing(cmb_ini, fwhm=np.deg2rad(beam)/60, lmax=lmax)
    # np.save(f'./{freq}.npy', cmb)
    hp.write_map(f'./{freq}.fits', cmb, overwrite=True)

