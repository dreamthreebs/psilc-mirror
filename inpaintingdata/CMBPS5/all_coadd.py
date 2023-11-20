import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

freq_list = [40, 95, 155, 215, 270]

for freq in freq_list:
    cmb = np.load(f'../CMB5/{freq}.npy')
    ps = hp.read_map(f'/sharefs/alicpt/users/zrzhang/allFreqPSMOutput/observations/AliCPT_uKCMB/{freq}GHz/group3_map_{freq}GHz.fits', field=(0,1,2))
    
    m = cmb + ps
    
    hp.write_map(f'./{freq}.fits', m, overwrite=True)

def check_map():
    for i, type_m in enumerate("TQU"):
        hp.mollview(m[i], title=f"{type_m}", norm="hist")
        plt.show()

# check_map()

def check_cl():

    lmax = 2000
    l = np.arange(lmax+1)
    cl = hp.anafast(m, lmax=lmax)
    print(f'{cl.shape=}')
    
    for i, type_cl in enumerate("TEB"):
        plt.semilogy(l*(l+1)*cl[i]/(2*np.pi), label=f"{type_cl}")
        plt.legend()
        plt.xlim(0,2000)
        plt.show()

# check_cl()



