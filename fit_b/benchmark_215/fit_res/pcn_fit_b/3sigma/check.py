import numpy as np

for rlz_idx in range(100):

    fuck = np.load(f'./mask_{rlz_idx}.npy')
    print(f'{fuck=}')
