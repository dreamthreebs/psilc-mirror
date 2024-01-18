import numpy as np
import healpy as hp
import os
import glob

from pathlib import Path

nside_out = 256

dir_name = './PSCMB'
path = glob.glob(os.path.join(dir_name, '*.npy'))
print(f'{path=}')
sorted_path = sorted(path, key=lambda x: int(Path(x).stem))
print(f'{sorted_path=}')

for p in sorted_path:
    freq = Path(p).stem
    print(f"{freq=}")

    m = np.load(p)
    print(f'{np.size(m)=}')
    ud_m = hp.ud_grade(m, nside_out=nside_out)
    if not os.path.exists(f'./{nside_out}/{dir_name}'):
        os.mkdir(f'./{nside_out}/{dir_name}')
    
    np.save(f'./{nside_out}/{dir_name}/{freq}.npy', ud_m)





