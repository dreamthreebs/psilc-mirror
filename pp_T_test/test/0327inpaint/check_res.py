import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

nside = 2048
lmax = 2000
l = np.arange(lmax+1)
npix = hp.nside2npix(nside)
beam = 17
bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=lmax)

ps_cmb_noise = hp.read_map('./fits_file/ps_cmb_noise.fits')
mask = hp.read_map('./fits_file/mask.fits')
cmb_noise = np.load('./cmb_noise.npy')
inpaint_res1 = hp.read_map('./fits_file/inpaint_50.fits')
inpaint_res = np.load('./diffuse_inpaint/diffuse_m.npy')
# inpaint_res1 = hp.read_map('./diffuse_inpaint/diffuse_m.npy')
inpaint_res2 = hp.read_map('./fits_file/inpaint_l_8000.fits')

# hp.mollview(ps_cmb_noise, title='ps_cmb_noise', xsize=4000)
# hp.mollview(mask, title='mask', xsize=4000)
hp.mollview(inpaint_res, title='inpaint_res', xsize=4000)
# hp.mollview(cmb_noise, title='cmb_noise', xsize=4000)
# hp.mollview(ps_cmb_noise * mask, title='ps_cmb_noise * mask', xsize=4000)
plt.show()

cl_masked_ps_cmb_noise = hp.anafast(ps_cmb_noise*mask, lmax=lmax)
cl_ps_cmb_noise = hp.anafast(ps_cmb_noise, lmax=lmax)
cl_cmb_noise = hp.anafast(cmb_noise, lmax=lmax)
cl_inpaint = hp.anafast(inpaint_res, lmax=lmax)

cl_inpaint1 = hp.anafast(inpaint_res1, lmax=lmax)
cl_inpaint2 = hp.anafast(inpaint_res2, lmax=lmax)


plt.plot(l*(l+1)*cl_masked_ps_cmb_noise/(2*np.pi)/bl**2, label='dl masked_ps_cmb_noise')
plt.plot(l*(l+1)*cl_ps_cmb_noise/(2*np.pi)/bl**2, label='dl ps_cmb_noise')
plt.plot(l*(l+1)*cl_cmb_noise/(2*np.pi)/bl**2, label='dl cmb_noise')
# plt.plot(l*(l+1)*cl_inpaint/(2*np.pi)/bl**2, label='dl inpaint_40_1.5')
plt.plot(l*(l+1)*cl_inpaint1/(2*np.pi)/bl**2, label='dl inpaint iter 50')
plt.plot(l*(l+1)*cl_inpaint2/(2*np.pi)/bl**2, label='dl inpaint lmax 8000')
plt.plot(l*(l+1)*cl_inpaint/(2*np.pi)/bl**2, label='dl inpaint diffuse')
plt.semilogy()
plt.legend()
plt.xlim(2,2000)
plt.ylim(1e2, 1e6)
plt.xlabel('$\\ell$')
plt.ylabel('$D_\\ell^{TT} [\mu K^2]$')
plt.show()




