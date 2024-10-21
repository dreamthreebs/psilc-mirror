import numpy as np

for i in range(200):
    slope = np.load(f'./{i}.npy')
    print(f'{slope=}')
