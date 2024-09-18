import numpy as np

for i in range(200):
    m = np.load(f'./{i}.npy')
    print(f'{m=}')

