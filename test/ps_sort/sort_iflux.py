import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pandas as pd
import os

from pathlib import Path
from scipy.io import readsav

freq = 40
data = readsav(f'/sharefs/alicpt/users/zrzhang/allFreqPSMOutput/skyinbands/AliCPT_uKCMB/{freq}GHz/strongirps_cat_{freq}GHz.sav', python_dict=True, verbose=True)

lon = data['comp']['lon'][0][0][0]
lat = data['comp']['lat'][0][0][0]
iflux = data['comp']['obs1'][0]['iflux'][0]
qflux = data['comp']['obs1'][0]['qflux'][0]
uflux = data['comp']['obs1'][0]['uflux'][0]

pflux = np.sqrt(qflux**2 + uflux**2)

dfir = pd.DataFrame({
    "lon": lon.astype('float64'),
    "lat": lat.astype('float64'),
    "iflux": iflux.astype('float64'),
    "qflux": qflux.astype('float64'),
    "uflux": uflux.astype('float64'),
    "pflux": pflux.astype('float64')
        }
        )

# dfir.to_csv(f'irps{freq}.csv', index=False)

data = readsav(f'/sharefs/alicpt/users/zrzhang/allFreqPSMOutput/skyinbands/AliCPT_uKCMB/{freq}GHz/strongradiops_cat_{freq}GHz.sav', python_dict=True, verbose=False)

lon = data['comp']['lon'][0][0][0]
lat = data['comp']['lat'][0][0][0]
iflux = data['comp']['obs1'][0]['iflux'][0]
qflux = data['comp']['obs1'][0]['qflux'][0]
uflux = data['comp']['obs1'][0]['uflux'][0]

pflux = np.sqrt(qflux**2 + uflux**2)

dfradio = pd.DataFrame({
    "lon": lon.astype('float64'),
    "lat": lat.astype('float64'),
    "iflux": iflux.astype('float64'),
    "qflux": qflux.astype('float64'),
    "uflux": uflux.astype('float64'),
    "pflux": pflux.astype('float64')
        }
        )

# dfradio.to_csv(f'radiops{freq}.csv', index=False)

df = pd.concat([dfir, dfradio], ignore_index=True, axis=0)

df1 = df.sort_values(by='iflux', ascending=False)

df2 = df1.reset_index()
df2.to_csv(f'./sort_by_iflux/{freq}.csv', index=True)










