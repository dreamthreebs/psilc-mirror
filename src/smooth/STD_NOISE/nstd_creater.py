import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import pandas as pd

df = pd.read_csv('../../../FGSim/FreqBand')
n_freq = len(df)
nside = 512
n_pix = hp.nside2npix(nside)

for i in range(n_freq):
    nstd = df.at[i,'nstd']
    freq = df.at[i,'freq']
    m = nstd * np.ones((3,n_pix))
    m[0] = nstd * np.ones((n_pix)) / np.sqrt(2)
    hp.mollview(m[0])
    hp.mollview(m[1]);plt.show()
    np.save(f'../../../FGSim/NSTDNORTH/{freq}.npy', m)



