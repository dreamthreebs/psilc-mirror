import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

freq = 40
m = hp.read_map(f'./{freq}.fits', field=0)
cmbps = hp.read_map(f'../../../../inpaintingdata/CMBPS5/{freq}.fits', field=0)

fsky = np.sum(m)/np.size(m)

print(f'{fsky=}')
hp.mollview(m, title='mask')
hp.mollview(m*cmbps, title='mask*cmbps', norm='hist')
hp.mollview(cmbps, title='cmbps', norm='hist')
plt.show()
