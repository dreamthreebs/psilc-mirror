import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pymaster as nmt


lmax=500
l = np.arange(lmax+1)
nside=512
bl_out = hp.gauss_beam(np.deg2rad(9)/60, lmax=2000, pol=True)
b = nmt.NmtBin.from_lmax_linear(lmax, 20, is_Dell=True)
ell_arr = b.get_effective_ells()
def calc_dl_from_scalar_map(scalar_map, apo_mask):
    scalar_field = nmt.NmtField(apo_mask, [scalar_map], beam=bl_out[:,2])
    dl = nmt.compute_full_master(scalar_field, scalar_field, b)
    return dl[0]

cl_theo = np.load('../../ebleakage/cmbdata/cmbcl3.npy')[:lmax+1,2]


m = np.load('./smcmbtest/30.npy')
true = np.load('../../../FGSim/CMB/270.npy')
print(f'{true.shape=}')
bin_mask = np.load('../../mask/north/BINMASKG.npy')
mask = np.load('../../mask/north/APOMASKC1_10.npy')
print(f'{mask.shape=}')
smmask = hp.smoothing(mask, fwhm=np.deg2rad(2))


true_b = hp.alm2map(hp.map2alm(true, lmax=lmax)[2], nside=nside) * bin_mask
res_b = true_b - m

print(f'{true_b.shape=}')
dl_true = calc_dl_from_scalar_map(scalar_map=true_b, apo_mask=mask)
dl_eblc = calc_dl_from_scalar_map(scalar_map=m, apo_mask=mask)
dl_res = calc_dl_from_scalar_map(scalar_map=res_b, apo_mask=mask)



# cl_true = hp.anafast(true_b, lmax=lmax)
# cl_eblc = hp.anafast(m*mask, lmax=lmax)
# cl_res = hp.anafast(m*mask- true_b, lmax=lmax)

# plt.semilogy(l*(l+1)*cl_true/(2*np.pi))
# plt.semilogy(l*(l+1)*cl_eblc/(2*np.pi))
# plt.semilogy(l*(l+1)*cl_res/(2*np.pi))
plt.semilogy(l*(l+1)*cl_theo/(2*np.pi), label='r=0.01')

plt.semilogy(ell_arr, dl_true, label='true' )
plt.semilogy(ell_arr, dl_eblc, label='after eblc' )
plt.semilogy(ell_arr, dl_res , label='res')
plt.legend()
plt.xlabel('$\\ell$')
plt.ylabel('$D_\\ell$')

plt.show()

