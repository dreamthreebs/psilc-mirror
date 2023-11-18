import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import readsav
import os
from pathlib import Path


def mask_ps(nside, freq, radius):

    mask = np.ones(hp.nside2npix(nside))
    data = readsav(f'/sharefs/alicpt/users/zrzhang/allFreqPSMOutput/skyinbands/AliCPT_uKCMB/{freq}GHz/strongirps_cat_{freq}GHz.sav', python_dict=True, verbose=False)

    lon = data['comp']['lon'][0][0][0]
    print(f'{lon.shape=}')
    lat = data['comp']['lat'][0][0][0]
    print(f'{lat.shape=}')

    for i in range(len(lon)):
        vec = hp.ang2vec(np.rad2deg(lon[i]), np.rad2deg(lat[i],), lonlat=True)
        ipix = hp.query_disc(nside, vec, radius=np.deg2rad(radius)/60)
        # print(f'{ipix=}')
        mask[ipix] = 0
    
    data = readsav(f'/sharefs/alicpt/users/zrzhang/allFreqPSMOutput/skyinbands/AliCPT_uKCMB/{freq}GHz/strongradiops_cat_{freq}GHz.sav', python_dict=True, verbose=False)
    
    lon = data['comp']['lon'][0][0][0]
    print(f'{lon.shape=}')
    lat = data['comp']['lat'][0][0][0]
    print(f'{lat.shape=}')
    
    for i in range(len(lon)):
        vec = hp.ang2vec(np.rad2deg(lon[i]), np.rad2deg(lat[i],), lonlat=True)
        ipix = hp.query_disc(nside, vec, radius=np.deg2rad(radius)/60)
        # print(f'{ipix=}')
        mask[ipix] = 0

    # hp.mollview(mask)
    # plt.show()

    return mask



if __name__ == '__main__':

    df = pd.read_csv('../../FGSim/FreqBand5')
    
    nside = 2048
    fold = 1.5
    directory = Path(f"./psmaskfits2048/{fold}")
    directory.mkdir(parents=True, exist_ok=True)

    for i in range(len(df)):
        freq = df.at[i, 'freq']
        beam = df.at[i, 'beam']
        print(f'{freq=}, {beam=}')
        radius = fold * beam
        print(f'{radius=}')
        mask = mask_ps(nside, freq, radius=radius)
        hp.write_map(f'./psmaskfits2048/{fold}/{freq}.fits', mask, overwrite=True)













