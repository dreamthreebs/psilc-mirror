import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import sys

bl_std_curl = np.load('../../smooth/BL/bl_std_curl.npy')
cmb = np.load('../../../FGSim/CMB/270.npy')

cl_cmb = hp.anafast(cmb, lmax=300)

cl1 = np.load('./nilc_cl1.npy')
cl2 = np.load('./nilc_cl6.npy')
cl3 = np.load('./nilc_cl7.npy')
# cl4 = np.load('./nilc_cl4.npy')
# cl5 = np.load('./nilc_cl5.npy')
fgres_cl1 = np.load('./nilc_fgres_cl1.npy')
fgres_cl2 = np.load('./nilc_fgres_cl6.npy')
fgres_cl3 = np.load('./nilc_fgres_cl7.npy')
# fgres_cl4 = np.load('./nilc_fgres_cl4.npy')
# fgres_cl5 = np.load('./nilc_fgres_cl5.npy')
noise_cl2 = np.load('./NOISE/nilc_noise_cl6avg.npy')
noise_cl3 = np.load('./lowNOISE/nilc_noise_cl7avg.npy')
# noise_cl4 = np.load('./nilc_noise_cl4.npy')
# noise_cl5 = np.load('./nilc_noise_cl5.npy')

print(cl1.shape)
l = np.arange(len(cl1))

fig, axs = plt.subplots(2, 1, sharex=True, height_ratios=[3,1])
axs[0].semilogy(l*(l+1)*cl_cmb[2]/(2*np.pi)/bl_std_curl**2, label='input', color='k')
axs[0].semilogy(l*(l+1)*(cl2-noise_cl2)/(2*np.pi)/bl_std_curl**2, label='no_PS_fg debias noise')
axs[0].semilogy(l*(l+1)*(cl3-noise_cl3)/(2*np.pi)/bl_std_curl**2, label='no_PS_fg debias lower noise')
axs[0].legend(fontsize=8, loc='upper left')
axs[0].set_ylabel('$D_\\ell$')
axs[0].set_ylim([5e-6,1e-1])
# axs[0].tick_params(axis='both', labelsize=8)
# axs[0].yaxis.set_label_coords(0.5, -0.2)

# Calculate Std Dev
std_dev_2_vs_1 = np.std((cl2-noise_cl2) - cl_cmb[2])
std_dev_3_vs_1 = np.std((cl3-noise_cl3) - cl_cmb[2])

# Second Plot
axs[1].semilogy(l, l*(l+1)*std_dev_2_vs_1/(2*np.pi)/bl_std_curl**2 )
axs[1].semilogy(l, l*(l+1)*std_dev_3_vs_1/(2*np.pi)/bl_std_curl**2  )
# axs[1].legend()
axs[1].set_ylabel('std of $D_\\ell$')
axs[1].set_ylim([1e-5,1e-1])
# axs[1].tick_params(axis='both', labelsize=8)
# axs[1].yaxis.set_label_coords(-0.1, 0.5)

plt.xlabel('$\\ell$')
plt.subplots_adjust(hspace=0)
plt.xlim(10,280)
plt.ylabel('$D_\\ell$')
plt.savefig(f'/afs/ihep.ac.cn/users/w/wangyiming25/tmp/20230912/noisedebiased_cl.png',dpi=300)
plt.show()

# plt.legend()
# plt.xlim(10,280)
# plt.title('foreground residual')
# plt.title('debiased power spectrum')
# plt.savefig(f'/afs/ihep.ac.cn/users/w/wangyiming25/tmp/20230907/testspsilc_fgres.png',dpi=300)

