import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('../../FGSim/FreqBand5')
fold = 0.8

for i in range(len(df)):
    freq = df.at[i, 'freq']
    beam = df.at[i, 'beam']
    print(f'{freq=}, {beam=}')
    m = np.load(f'./{freq}.npy')
    mask = hp.read_map(f'./psmaskfits/{fold}/{freq}.fits', field=0)

    q = m[1]
    u = m[2]
    p2 = q**2 +u**2
    hp.mollview(mask)
    hp.mollview(p2, norm='hist', min=-10, max=10)
    hp.mollview(p2 * mask, norm='hist')
    plt.show()
