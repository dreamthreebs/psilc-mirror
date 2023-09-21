import numpy as np
import matplotlib.pyplot as plt
import healpy as hp


lmax = 300
nside = 512
cmb = np.load('../../FGSim/CMB/270.npy')
bin_mask = np.load('../mask/north/BINMASKG.npy')
apo_mask = np.load('../mask/north/APOMASKC1_1.npy')
almfull = hp.map2alm(cmb, lmax=lmax)
B_cmb = hp.alm2map(almfull[2], nside=nside) * apo_mask

p_cmb = cmb * apo_mask
alm_cmb = hp.map2alm(p_cmb, lmax=lmax)

alm_T = alm_cmb[0]
alm_E = alm_cmb[1]
alm_B = alm_cmb[2]

E_family = hp.alm2map([alm_T,alm_E,np.zeros_like(alm_T)], nside=nside) * apo_mask # TQU E family maps
B_family = hp.alm2map([alm_T,np.zeros_like(alm_T),alm_B], nside=nside) * apo_mask # contaminated B
EBleakage = hp.map2alm(E_family, lmax=lmax)
B_family_temp = hp.alm2map([EBleakage[0],np.zeros_like(alm_T),EBleakage[2]], nside=nside) * apo_mask

# coeffs = np.polyfit(hp.pixelfunc.ma(B_family_temp[1:2], badval=0).flatten(), hp.pixelfunc.ma(B_family[1:2], badval=0).flatten(), 1)
coeffs = np.polyfit(B_family_temp[1:2].flatten()[np.nonzero(B_family_temp[1:2].flatten())], B_family[1:2].flatten()[np.nonzero(B_family[1:2].flatten())], 1)
slope, intercept = coeffs

print(f"Slope: {slope}, Intercept: {intercept}")

hp.mollview(B_family[1])
plt.plot(B_family[1:2].flatten())
plt.plot(B_family_temp[1:2].flatten())
plt.show()

# B_family_cleaned = B_family - slope * B_family_temp
# cleaned = hp.alm2map(hp.map2alm(B_family_cleaned, lmax=lmax)[2], nside=nside) * apo_mask
# corrupted = hp.alm2map(hp.map2alm(B_family, lmax=lmax)[2], nside=nside) * apo_mask

# # hp.mollview(p_cmb[0]);plt.show()
# hp.orthview(cleaned, half_sky=True, rot=[100,50,0], title='cleaned', min=-0.5, max=0.5)
# hp.orthview(corrupted, half_sky=True, rot=[100,50,0], title='corrupted', min=-0.5, max=0.5)
# hp.orthview(B_cmb, half_sky=True, rot=[100,50,0], title='input', min=-0.5, max=0.5);plt.show()
# hp.orthview(cleaned - B_cmb, half_sky=True, rot=[100,50,0], title='residual', min=-0.5, max=0.5);plt.show()


