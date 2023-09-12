import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import sys

bl_std_curl = np.load('../smooth/BL/bl_std_curl.npy')
cmb = np.load('../../FGSim/CMB/270.npy')

cl_cmb = hp.anafast(cmb, lmax=300)

pilc_cl = np.load('./PILCRESULT/pilc_cl2.npy')
pilc_fgres_cl = np.load('./PILCRESULT/pilc_fgres_cl2.npy')
pilc_noise_cl = np.load('./PILCRESULT/pilc_noise_cl2.npy')
pilc_noise_cl1 = np.load('./PILCRESULT/pilc_noise_avg2.npy')

hilc_cl = np.load('./HILCRESULT/hilc_cl2.npy')
hilc_fgres_cl = np.load('./HILCRESULT/hilc_fgres_cl2.npy')
hilc_noise_cl = np.load('./HILCRESULT/hilc_noise_cl2.npy')
hilc_noise_cl1 = np.load('./HILCRESULT/hilc_noise_avg2.npy')

nilc_cl = np.load('./FULL_B/nilc_cl6.npy')
nilc_fgres_cl = np.load('./FULL_B/nilc_fgres_cl6.npy')
nilc_noise_cl = np.load('./FULL_B/NOISE/nilc_noise_cl6avg.npy')

l = np.arange(len(hilc_cl))
plt.semilogy(l*(l+1)*cl_cmb[2]/(2*np.pi)/bl_std_curl**2,label='input')

# plt.semilogy(l*(l+1)*nilc_cl/(2*np.pi)/bl_std_curl**2,label='nilc')
# plt.semilogy(l*(l+1)*hilc_cl/(2*np.pi)/bl_std_curl**2,label='hilc')
# plt.semilogy(l*(l+1)*pilc_cl/(2*np.pi)/bl_std_curl**2,label='pilc')

# plt.semilogy(l*(l+1)*nilc_fgres_cl/(2*np.pi)/bl_std_curl**2,label='nilc fgres')
# plt.semilogy(l*(l+1)*hilc_fgres_cl/(2*np.pi)/bl_std_curl**2,label='hilc fgres')
# plt.semilogy(l*(l+1)*pilc_fgres_cl/(2*np.pi)/bl_std_curl**2,label='pilc fgres')

# plt.semilogy(l*(l+1)*(nilc_cl-nilc_noise_cl)/(2*np.pi)/bl_std_curl**2,label='nilc debias')
# plt.semilogy(l*(l+1)*(hilc_cl-hilc_noise_cl)/(2*np.pi)/bl_std_curl**2,label='hilc debias')
# plt.semilogy(l*(l+1)*(hilc_cl-hilc_noise_cl1)/(2*np.pi)/bl_std_curl**2,label='hilc debias1')
plt.semilogy(l*(l+1)*(pilc_cl-pilc_noise_cl)/(2*np.pi)/bl_std_curl**2,label='pilc debias')
plt.semilogy(l*(l+1)*(pilc_cl-pilc_noise_cl1)/(2*np.pi)/bl_std_curl**2,label='pilc debias1')



plt.legend()
plt.ylim(1e-6,1e-1)
plt.xlim(10,280)
# plt.title('foreground residual')
# plt.title('debiased power spectrum')
plt.xlabel('$\\ell$')
plt.ylabel('$D_\\ell$')
# plt.savefig(f'/afs/ihep.ac.cn/users/w/wangyiming25/tmp/20230907/testspsilc_fgres.png',dpi=300)
# plt.savefig(f'/afs/ihep.ac.cn/users/w/wangyiming25/tmp/20230907/testpsilc_debias.png',dpi=300)
plt.show()

