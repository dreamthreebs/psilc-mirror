import numpy as np

from pathlib import Path

path_cn = Path(f'./cn')
path_cn.mkdir(exist_ok=True, parents=True)

for rlz_idx in range(1000):
    print(f'{rlz_idx=}')
    cmb = np.load(f'./cmb/{rlz_idx}.npy')
    noise = np.load(f'./noise/{rlz_idx}.npy')
    cn = cmb + noise

    np.save(f'./cn/{rlz_idx}.npy', cn)



