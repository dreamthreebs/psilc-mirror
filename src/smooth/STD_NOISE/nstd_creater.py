import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import pandas as pd

df = pd.read_csv('../../../FGSim/FreqBand')
n_freq = len(df)
nside=512
n_pix = hp.nside2npix(nside)

for i in range(n_freq):
    nstd = df.at[i,'nstd']
    freq = df.at[i,'freq']
    m = nstd * np.ones((3,n_pix))
    hp.mollview(m[0]);plt.show()
    np.save('../../../FGSim/NSTDNORTH/{freq}.npy', nstd)



