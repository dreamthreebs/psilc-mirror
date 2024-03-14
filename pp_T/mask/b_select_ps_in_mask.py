import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

for freq in [30,40,85,95,145,155,215,270]:
    # freq = 30
    nside = 2048
    mask = np.load('../../src/mask/north/BINMASKG2048.npy')
    df = pd.read_csv(f'./ps_csv/{freq}.csv')
    
    def in_or_out_mask(row):
        lon = row['lon']
        lat = row['lat']
        ipix = hp.ang2pix(nside=nside, theta=np.rad2deg(lon), phi=np.rad2deg(lat), lonlat=True)
        if mask[ipix] == 1.0:
            return True
        elif mask[ipix] == 0.0:
            return False
        else:
            raise ValueError('mask data type should be float!')
    
    condition = df.apply(in_or_out_mask, axis=1)
    
    filtered_data = df[condition]
    filtered_data.reset_index(drop=True, inplace=True)
    print(f'{filtered_data=}')
    
    mask_csv_path = Path('./mask_csv')
    mask_csv_path.mkdir(parents=True, exist_ok=True)
    
    filtered_data.to_csv(mask_csv_path / Path(f'{freq}.csv'), index=True)

