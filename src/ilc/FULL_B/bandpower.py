import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import pymaster as nmt

nside = 128
cl_type = 'B'

# Initialize binning scheme with bandpowers of constant width
# (4 multipoles per bin)
bin1 = nmt.NmtBin.from_nside_linear(nside, 50)

ell_eff = bin1.get_effective_ells()
print(f'{ell_eff.shape = }')

ell_arr = np.arange(384)
print(f'{ell_arr.shape = }')

bl_std_curl = np.load('../../smooth/BL/bl_std_curl.npy')
print(f'{bl_std_curl.shape=}')
bl = np.pad(bl_std_curl, (0, 384 - len(bl_std_curl)), 'constant')
bl1_binned = bin1.bin_cell(np.array([bl]))

cmb = np.load('../../../FGSim/CMB/270.npy')

cl_cmb = hp.anafast(cmb, lmax=383)[2]
cl_cmb_binned = bin1.bin_cell(np.array([cl_cmb]))

cl1 = np.load(f'./nilc_cl6.npy')
print(f'{cl1.shape = }')

cl1 = np.pad(cl1, (0, 384 - len(cl1)), 'constant')
cl1_binned = bin1.bin_cell(np.array([cl1]))

cl2 = np.load(f'./NOISE/nilc_noise_cl6avg.npy')

cl2 = np.pad(cl2, (0, 384 - len(cl2)), 'constant')
cl2_binned = bin1.bin_cell(np.array([cl2]))




# Plot all to see differences
# plt.plot(ell_arr, cl/500, 'black',
         # label='Theory $C_\\ell$')

plt.semilogy(ell_eff, ell_eff*(ell_eff+1)*cl_cmb_binned[0]/(2*np.pi)/bl1_binned[0]**2, 'g-',
         label='theory Binned $C_\\ell$')

plt.semilogy(ell_eff, ell_eff*(ell_eff+1)*(cl1_binned[0]-cl2_binned[0])/(2*np.pi)/bl1_binned[0]**2,
         label='Binned $C_\\ell$')

# plt.loglog(ell_eff, np.abs(cl2_binned[0]/cl_cmb_binned[0]*100),
#          label='Binned1  $C_\\ell$')

# plt.loglog(ell_eff, np.abs(cl3_binned[0]/cl_cmb_binned[0]*100),
#          label='Binned2  $C_\\ell$')


# plt.loglog()
plt.ylim(1e-6,1e-1)
plt.legend(loc='upper right', frameon=False)
plt.xlabel('ell')
# plt.ylabel('cl_sd/(1/100)cl_cmb')
plt.show()



