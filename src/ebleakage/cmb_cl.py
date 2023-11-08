import sys, platform, os
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import camb
from camb import model, initialpower
camb_path = os.path.dirname(camb.__file__)
print(f'{camb_path}')
print('Using CAMB %s installed at %s'%(camb.__version__,os.path.dirname(camb.__file__)))
pars = camb.read_ini(os.path.join(camb_path,'inifiles','planck_2018_acc.ini'))
pars.set_for_lmax(4000, lens_potential_accuracy=2);

results = camb.get_results(pars)
powers =results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=True)
for name in powers: print(name)

totcl = powers['total']
print(f'{totcl.shape}')
unlensedcl = powers['unlensed_total']

l = np.arange(totcl.shape[0])
# print(f'{totcl[:,2]}')
print(f'{unlensedcl.shape}')


# totcl[:,2] = totcl[:,2]
for index, flag in enumerate(['TT', 'EE', 'BB']):
    plt.loglog(l*(l*1)*totcl[:,index]/(2*np.pi),label=f'{flag} lensed')
    # plt.loglog(l*(l*1)*unlensedcl[:,index]/(2*np.pi),label=f'{flag} unlensed')

plt.legend()
plt.xlabel('l')
plt.ylabel('Dl')

plt.show()

np.save('./cmbdata/cmbcl4.npy', totcl)


