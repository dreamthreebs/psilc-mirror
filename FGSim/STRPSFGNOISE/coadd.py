import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../FreqBand5')

for i in range(len(df)):
    freq = df.at[i, 'freq']
    print(f'{freq=}')

    strongps = hp.read_map(f'/sharefs/alicpt/users/zrzhang/allFreqPSMOutput/observations/AliCPT_uKCMB/{freq}GHz/group3_map_{freq}GHz.fits', field=(0,1,2))
    print(f'{hp.get_nside(strongps)=}')

    diffusefg = hp.read_map(f'/sharefs/alicpt/users/zrzhang/allFreqPSMOutput/observations/AliCPT_uKCMB/{freq}GHz/group1_map_{freq}GHz.fits', field=(0,1,2))

    cmb = np.load(f'../../inpaintingdata/CMB8/{freq}.npy')
    # hp.mollview(ps[0], norm='hist', title='ps2048')
    # plt.show()

    # hp.mollview(ps512[0], norm='hist', title='ps512')
    # plt.show()

    nstd = np.load(f'../NSTDNORTH/2048/{freq}.npy')
    n_pix_nstd = hp.get_map_size(nstd[0])

    noise = nstd * np.random.normal(0,1,(3,n_pix_nstd))

    # hp.mollview(noise, title=f'noise at {freq=}')
    # plt.show()

    syn_map = strongps + diffusefg + noise

    hp.mollview(syn_map[0], title=f'{freq=}', norm='hist')
    plt.show()
    
    np.save(f'./{freq}.npy', syn_map)


