import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pymaster as nmt

nside=1024
mask = hp.read_map('/sharefs/alicpt/users/chenwz/Testarea/PARTIAL_SKY/AliCPT_20uKcut150_C_1024.fits', field=0)
sm_mask = hp.read_map('/sharefs/alicpt/users/chenwz/Testarea/PARTIAL_SKY/mask_1024_Sm.fits',field=0)
m = hp.read_map('/sharefs/alicpt/users/chenwz/Testarea/PARTIAL_SKY/lensed_1024_map.fits',field=(0,1,2))
print(f'{mask.shape=}')
print(f'{m.shape=}')

lmax=1000
l = np.arange(lmax+1)
cl = hp.anafast(m, lmax=lmax)


# hp.mollview(mask)
# hp.mollview(sm_mask)
# hp.mollview(m[1])
# plt.show()

b = nmt.NmtBin.from_nside_linear(nside, 4, is_Dell=True)
ell_arr = b.get_effective_ells()

f2_np = nmt.NmtField(sm_mask, [m[1], m[2]] )
f2_yp = nmt.NmtField(sm_mask, [m[1], m[2]], purify_b=True )

cl_22_np = nmt.compute_full_master(f2_np, f2_np, b)
cl_22_yp = nmt.compute_full_master(f2_yp, f2_yp, b)

plt.plot(ell_arr, cl_22_np[3],  label='BB no purify')
plt.plot(ell_arr, cl_22_yp[3],  label='BB purify')
plt.plot(l, l*(l+1)*cl[2]/(2 * np.pi), label='BB full sky')
plt.loglog()
plt.xlabel('$\\ell$', fontsize=16)
plt.ylabel('$C_\\ell$', fontsize=16)
plt.legend(loc='upper right', ncol=2, labelspacing=0.1)
plt.show()
