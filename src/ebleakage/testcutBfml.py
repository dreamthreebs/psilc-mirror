import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

lmax = 2000
l = np.arange(lmax+1)
nside = 2048


m = np.load('./cmbdata/cmbtqunoB20483.npy')
bin_mask = np.load('./circle_mask2048.npy')
apo_mask = np.load('./apo_circle_mask2048C1_12.npy')
fsky = np.sum(bin_mask)/np.size(bin_mask)
fsky_apo = np.sum(apo_mask)/np.size(apo_mask)
print(f'{m.shape}')

full_cl = hp.anafast(m, lmax=lmax)

cut_alms = hp.map2alm(m * bin_mask, lmax=lmax)
cut_almT, cut_almE, cut_almB = [alm for alm in cut_alms]

fml_B = hp.alm2map([cut_almT, np.zeros_like(cut_almT), cut_almB], nside=nside)
fml_E = hp.alm2map([cut_almT, cut_almE, np.zeros_like(cut_almT)], nside=nside)
masked_fml_E = fml_E * bin_mask
template_fml_B = hp.alm2map([cut_almT, np.zeros_like(cut_almT), hp.map2alm(masked_fml_E, lmax=lmax)[2]], nside=nside) * bin_mask

masked_fml_B = fml_B * bin_mask

coeffs = np.polyfit(template_fml_B[1:2].flatten(), masked_fml_B[1:2].flatten(), 1)
slope, intercept = coeffs
print(f"Slope: {slope}, Intercept: {intercept}")
cleaned_QU = masked_fml_B - slope * template_fml_B

iter_1 = hp.alm2map([cut_almT, np.zeros_like(cut_almT), hp.map2alm(cleaned_QU, lmax=lmax)[2]], nside=nside) * bin_mask
iter_2 = hp.alm2map([cut_almT, np.zeros_like(cut_almT), hp.map2alm(iter_1, lmax=lmax)[2]], nside=nside) * bin_mask
iter_3 = hp.alm2map([cut_almT, np.zeros_like(cut_almT), hp.map2alm(iter_2, lmax=lmax)[2]], nside=nside) * bin_mask




cut_cl = hp.anafast(masked_fml_B, lmax=lmax)
cln_cl = hp.anafast(cleaned_QU, lmax=lmax)
iter_1_cl = hp.anafast(iter_1, lmax=lmax)
iter_2_cl = hp.anafast(iter_2, lmax=lmax)
iter_3_cl = hp.anafast(iter_3, lmax=lmax)


plt.loglog(l*(l+1)*full_cl[2]/(2*np.pi))
plt.loglog(l*(l+1)*cut_cl[2]/(2*np.pi)/fsky)
plt.loglog(l*(l+1)*cln_cl[2]/(2*np.pi)/fsky)
plt.loglog(l*(l+1)*iter_1_cl[2]/(2*np.pi)/fsky)
plt.loglog(l*(l+1)*iter_2_cl[2]/(2*np.pi)/fsky)
plt.loglog(l*(l+1)*iter_3_cl[2]/(2*np.pi)/fsky)
plt.show()



