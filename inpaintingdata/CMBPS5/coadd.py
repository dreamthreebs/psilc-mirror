import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

cmb = np.load('../CMB5/40.npy')
ps = hp.read_map('/sharefs/alicpt/users/zrzhang/allFreqPSMOutput/observations/AliCPT_uKCMB/40GHz/group3_map_40GHz.fits', field=(0,1,2))

m = cmb + ps

hp.write_map('./40.fits', m, overwrite=True)

def check_map():
    for i, type_m in enumerate("TQU"):
        hp.mollview(m[i], title=f"{type_m}", norm="hist")
        plt.show()

# check_map()

lmax = 2000
l = np.arange(lmax+1)
cl = hp.anafast(m, lmax=lmax)
print(f'{cl.shape=}')

for i, type_cl in enumerate("TEB"):
    plt.semilogy(l*(l+1)*cl[i]/(2*np.pi), label=f"{type_cl}")
    plt.legend()
    plt.xlim(0,1000)
    plt.show()




