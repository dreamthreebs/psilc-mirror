import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

nstd = np.load('../../../FGSim/NSTDNORTH/2048/155.npy')
print(f'{nstd.shape=}')
n_rlz = 1000

for i in range(500, 1000):
    print(f'{i=}')
    noise = nstd * np.random.normal(0, 1, size=(nstd.shape[0],nstd.shape[1]))
    np.save(f'./155/{i}.npy', noise)
    # print(f'{noise.shape=}')

# hp.mollview(noise[0])
# plt.show()



