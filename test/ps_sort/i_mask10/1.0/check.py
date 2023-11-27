import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

freq = 30
m = hp.read_map(f'./{freq}.fits', field=0)

fsky = np.sum(m)/np.size(m)

print(f'{fsky=}')
hp.mollview(m)
plt.show()
