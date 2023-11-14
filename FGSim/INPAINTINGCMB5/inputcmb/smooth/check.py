import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

m40 = np.load('../../../../FG5/strongps/40.npy')[0]
hp.mollview(np.abs(m40), min=0.1, max=1000, title='m40')
# plt.show()

for freq in [40, 95, 155, 215, 270]:
    m = hp.read_map(f'./{freq}.fits', field=0)
    hp.mollview(np.abs(m), min=0.1, max=1000)
    plt.show()

