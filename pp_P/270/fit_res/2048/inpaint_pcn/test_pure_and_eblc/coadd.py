import numpy as np

for rlz_idx in range(100):
    print(f'{rlz_idx=}')
    cmb = np.load(f'./cmb/{rlz_idx}.npy')
    noise = np.load(f'./noise/{rlz_idx}.npy')
    cn = cmb + noise

    np.save(f'./cn/{rlz_idx}.npy', cn)
