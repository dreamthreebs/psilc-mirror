import numpy as np

for i in range(0,136):
    num_ps = np.load(f'./CMBFGNOISE/1.5/idx_{i}/norm_beam.npy')
    print(f'{i=}, {num_ps=}')

