import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pandas as pd

lmax = 1750
l = np.arange(lmax+1)
nside = 2048

freq = 95
index = 1
df = pd.read_csv('../../FGSim/FreqBand5')
m = hp.read_map(f'../CMBPS5/{freq}.fits', field=(0,1,2))

beam = df.at[index, 'beam']
print(f'{beam=}')
beam_std = 11
print(f'{beam_std=}')
bl_std = hp.gauss_beam(fwhm=np.deg2rad(beam_std)/60, lmax=lmax, pol=True)
bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax, pol=True)
div = bl/bl_std
print(f'{div=}')


alm_T = hp.almxfl(hp.map2alm(m, lmax=lmax)[0], bl_std[:,0]/bl[:,0])
alm_E = hp.almxfl(hp.map2alm(m, lmax=lmax)[1], bl_std[:,1]/bl[:,1])
alm_B = hp.almxfl(hp.map2alm(m, lmax=lmax)[2], bl_std[:,2]/bl[:,2])

sm = hp.alm2map([alm_T, alm_E, alm_B], nside=nside)

hp.write_map(f"./{beam_std}arcmin/{freq}.fits", sm[0], overwrite=True)

# for i, type_m in enumerate("TQU"):
#     # hp.gnomview(sm[i], title=f'{type_m}')
#     hp.mollview(sm[i], title=f'{type_m}', norm="hist")
# plt.show()



# cl = hp.anafast(sm, lmax=lmax)
# for i, type_m in enumerate("TEB"):
#     plt.semilogy(l*(l+1)*cl[i]/(2*np.pi), label=f'{type_m}')
# plt.legend()
# plt.show()

