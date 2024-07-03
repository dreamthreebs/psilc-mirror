import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../../FGSim/FreqBand')

freq = df.at[6, 'freq']
beam = df.at[6, 'beam']
map_depth = df.at[6, 'mapdepth']
print(f'{freq=}, {beam=}, {map_depth=}')

nside = 512
npix = hp.nside2npix(nside)
lmax = 1500
l = np.arange(lmax+1)

bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax, pol=True)[:,2]
print(f'{bl.shape=}')
nl = (map_depth/bl)**2 / 3437.748**2

nstd = np.load('../../FGSim/NSTDNORTH/215.npy')

noise = nstd * np.random.normal(loc=0, scale=1, size=(3, npix))
nl_calc = hp.anafast(noise, lmax=lmax)


plt.loglog(l, nl, label='th')
plt.loglog(l, nl_calc[2]/bl**2, label='exp')
plt.show()

