import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import pandas as pd

df = pd.read_csv('../../FGSim/FreqBand')
n_freq = len(df)
nside = 512
lmax = 300

for i in range(n_freq):
    nstd = df.at[i,'nstd']
    freq = df.at[i,'freq']
    print(f'{freq=},{nstd=}')
    m_noi = nstd * np.random.normal(0,1,(hp.nside2npix(nside)))
    hp.mollview(m_noi,title=f'{freq=},{nstd=}')
    plt.show()

