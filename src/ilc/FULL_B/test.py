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
cl4 = np.load('./nilc_cl8.npy')
cl5 = np.load('./nilc_cl9.npy')
cl6 = np.load('./nilc_cl10.npy')
fgres_cl1 = np.load('./nilc_fgres_cl1.npy')
fgres_cl2 = np.load('./nilc_fgres_cl6.npy')
fgres_cl3 = np.load('./nilc_fgres_cl7.npy')
fgres_cl4 = np.load('./nilc_fgres_cl8.npy')
fgres_cl5 = np.load('./nilc_fgres_cl9.npy')
fgres_cl6 = np.load('./nilc_fgres_cl10.npy')

noise_cl2 = np.load('./nilc_noise_cl6.npy')
noise_cl3 = np.load('./nilc_noise_cl7.npy')
noise_cl4 = np.load('./NOISE/nilc_noise_cl6avg.npy')
noise_cl5 = np.load('./lowNOISE/nilc_noise_cl7avg.npy')
noise_cl6 = np.load('./PSNOISE/nilc_noise_cl8avg.npy')
noise_cl7 = np.load('./PSlowNOISE/nilc_noise_cl9avg.npy')

print(cl1.shape)
l = np.arange(len(cl1))
plt.semilogy(l*(l+1)*cl_cmb[2]/(2*np.pi)/bl_std_curl**2,label='input',color='k')

# plt.semilogy(l*(l+1)*cl1/(2*np.pi)/bl_std_curl**2,label='no_PS_fg no_noise NILC')
# plt.semilogy(l*(l+1)*cl6/(2*np.pi)/bl_std_curl**2,label='PS_fg no_noise NILC')
# plt.semilogy(l*(l+1)*cl3/(2*np.pi)/bl_std_curl**2,label='no_PS_fg with_lower_noise NILC')

# plt.semilogy(l*(l+1)*cl4/(2*np.pi)/bl_std_curl**2,label='nilc')
# plt.semilogy(l*(l+1)*cl5/(2*np.pi)/bl_std_curl**2,label='nilc low')

# plt.semilogy(l*(l+1)*fgres_cl1/(2*np.pi)/bl_std_curl**2,label='no_PS_fg no_noise NILC')
plt.semilogy(l*(l+1)*fgres_cl6/(2*np.pi)/bl_std_curl**2,label='PS_fg no_noise NILC')
# plt.semilogy(l*(l+1)*fgres_cl2/(2*np.pi)/bl_std_curl**2,label='no_PS_fg with_noise NILC')
# plt.semilogy(l*(l+1)*fgres_cl3/(2*np.pi)/bl_std_curl**2,label='no_PS_fg with_lower_noise NILC')
plt.semilogy(l*(l+1)*fgres_cl4/(2*np.pi)/bl_std_curl**2,label='PS_fg with_noise NILC')
plt.semilogy(l*(l+1)*fgres_cl5/(2*np.pi)/bl_std_curl**2,label='PS_fg with_lower_noise NILC')

# plt.semilogy(l*(l+1)*fgres_cl4/(2*np.pi)/bl_std_curl**2,label='fgres')
# plt.semilogy(l*(l+1)*fgres_cl5/(2*np.pi)/bl_std_curl**2,label='fgres')

# plt.plot(l*(l+1)*cl1/(2*np.pi)/bl_std_curl**2,label='no_PS_fg no_noise NILC')
# plt.plot(l*(l+1)*(cl2-noise_cl2)/(2*np.pi)/bl_std_curl**2,label='no_PS_fg debias noise ')

# plt.semilogy(l*(l+1)*fgres_cl4/(2*np.pi)/bl_std_curl**2,label='fgres')
# plt.semilogy(l*(l+1)*fgres_cl5/(2*np.pi)/bl_std_curl**2,label='fgres')

# plt.plot(l*(l+1)*cl1/(2*np.pi)/bl_std_curl**2,label='no_PS_fg no_noise NILC')
# plt.plot(l*(l+1)*(cl2-noise_cl2)/(2*np.pi)/bl_std_curl**2,label='no_PS_fg debias noise ')
# plt.plot(l*(l+1)*(cl3-noise_cl3)/(2*np.pi)/bl_std_curl**2,label='no_PS_fg debias lower noise')

# plt.plot(l*(l+1)*(cl2-noise_cl4)/(2*np.pi)/bl_std_curl**2,label='no_PS_fg debias noise ')
# plt.plot(l*(l+1)*(cl3-noise_cl5)/(2*np.pi)/bl_std_curl**2,label='no_PS_fg debias lower noise')
# plt.plot(l*(l+1)*(cl4-noise_cl6)/(2*np.pi)/bl_std_curl**2,label='PS_fg debias noise')
# plt.plot(l*(l+1)*(cl5-noise_cl7)/(2*np.pi)/bl_std_curl**2,label='PS_fg debias lower noise')


plt.legend()
plt.ylim(1e-6,1e-1)
plt.xlim(10,280)
plt.title('foreground residual')
# plt.title('debiased power spectrum')
plt.xlabel('$\\ell$')
plt.ylabel('$D_\\ell$')
plt.savefig(f'/afs/ihep.ac.cn/users/w/wangyiming25/tmp/20230912/psilc_fgres.png',dpi=300)
# plt.savefig(f'/afs/ihep.ac.cn/users/w/wangyiming25/tmp/20230912/psilc_debias.png',dpi=300)
plt.show()

