import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

lmax = 750
l = np.arange(lmax+1)
nside = 2048

m = hp.read_map('../CMBPS5/40.fits', field=(0,1,2))

bl_std = hp.gauss_beam(fwhm=np.deg2rad(30)/60, lmax=lmax, pol=True)
bl_40 = hp.gauss_beam(fwhm=np.deg2rad(63)/60, lmax=lmax, pol=True)


alm_T = hp.almxfl(hp.map2alm(m, lmax=lmax)[0], bl_std[:,0]/bl_40[:,0])
alm_E = hp.almxfl(hp.map2alm(m, lmax=lmax)[1], bl_std[:,1]/bl_40[:,1])
alm_B = hp.almxfl(hp.map2alm(m, lmax=lmax)[2], bl_std[:,2]/bl_40[:,2])

sm = hp.alm2map([alm_T, alm_E, alm_B], nside=nside)

hp.write_map("40.fits", sm[0], overwrite=True)

# for i, type_m in enumerate("TQU"):
#     # hp.gnomview(sm[i], title=f'{type_m}')
#     hp.mollview(sm[i], title=f'{type_m}', norm="hist")
# plt.show()



# cl = hp.anafast(sm, lmax=lmax)
# for i, type_m in enumerate("TEB"):
#     plt.semilogy(l*(l+1)*cl[i]/(2*np.pi), label=f'{type_m}')
# plt.legend()
# plt.show()

