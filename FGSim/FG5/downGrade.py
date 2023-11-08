import numpy as np
import healpy as hp
import json
import pandas as pd

nside_out = 512

df = pd.read_csv('../FreqBand')

for freq in df['freqband']:
    print(freq)
    m = hp.read_map(f'/home/zhangzirui/MYPSM/AliCPT_uKCMB_8Bands/{freq}GHz/group4_map_{freq}GHz.fits', field=(0,1,2))
    mout = hp.ud_grade(m, nside_out)
    np.save(f'{freq}.npy', mout.astype(np.float32))
