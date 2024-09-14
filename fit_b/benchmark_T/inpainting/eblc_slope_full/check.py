import numpy as np

for i in range(30):
    slope = np.load(f'./{i}.npy')
    print(f'{slope=}')

