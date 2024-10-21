import numpy as np

for i in range(200):
    m1 = np.load(f'./eblc_slope_cmb/{i}.npy')
    m2 = np.load(f'./eblc_slope_pcfn/{i}.npy')
    print(f'{m1=}, {m2=}')

