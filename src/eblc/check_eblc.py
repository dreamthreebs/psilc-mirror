from eblc_base import EBLeakageCorrection

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

def check_eblc():
    m = np.load('../ebleakage/cmbdata/cmbtqunoB20482.npy')
    mask = np.load('../ebleakage/circle_mask2048.npy')
    lmax = 4000
    nside = 2048
    obj = EBLeakageCorrection(m, lmax, nside, mask=mask, post_mask=mask, method='zzr')
    crt_b, tmp_b, cln_b = obj.run_eblc()
    hp.orthview(hp.ma(crt_b, badval=0), half_sky=True,sub=(1,3,1), title='corrupted', min=-0.3, max=0.3, cmap='jet', badcolor='white')
    hp.orthview(hp.ma(tmp_b, badval=0), half_sky=True,sub=(1,3,2), title='template', min=-0.3, max=0.3, cmap='jet', badcolor='white')
    hp.orthview(10*hp.ma(cln_b, badval=0), half_sky=True,sub=(1,3,3), title='residual', min=-0.3, max=0.3, cmap='jet', badcolor='white')
    plt.show()

def check_eblc_with_smooth():
    lmax = 500
    nside = 2048

    m = np.load('../../FGSim/CMB/270.npy')
    bl = hp.gauss_beam(np.deg2rad(9)/60, lmax=500, pol=True)

    # bl_curl = np.load('../smooth/BL/bl_std_curl.npy')
    # bl_temp = np.load('../smooth/BL/bl_std_temp.npy')
    # bl_grad = np.load('../smooth/BL/bl_std_grad.npy')

    alms = hp.map2alm(m, lmax=lmax)
    almT, almE, almB = [alm for alm in alms]
    almT = hp.almxfl(almT, 1/bl[:,0])
    almE = hp.almxfl(almE, 1/bl[:,1])
    almB = hp.almxfl(almB, 1/bl[:,2])
    m = hp.alm2map([almT, almE, almB], nside=nside)

    hp.mollview(m[0])
    plt.show()

    bin_mask = np.load('../mask/north/BINMASKG2048.npy')
    apo_mask = np.load('../mask/north/APOMASK2048C1_5.npy')
    fsky = np.sum(bin_mask)/np.size(bin_mask)
    apo_fsky = np.sum(apo_mask)/np.size(apo_mask)

    obj = EBLeakageCorrection(m, lmax, nside, mask=bin_mask, post_mask=apo_mask, method='zzr')
    crt_b, tmp_b, cln_b = obj.run_eblc()
    hp.orthview(hp.ma(crt_b, badval=0), rot=[100,50,0], half_sky=True,sub=(1,3,1), title='corrupted', min=-0.6, max=0.6, cmap='jet', badcolor='white')
    hp.orthview(hp.ma(tmp_b, badval=0), rot=[100,50,0],  half_sky=True,sub=(1,3,2), title='template', min=-0.6, max=0.6, cmap='jet', badcolor='white')
    hp.orthview(hp.ma(cln_b, badval=0), rot=[100,50,0],  half_sky=True,sub=(1,3,3), title='cleaned', min=-0.6, max=0.6, cmap='jet', badcolor='white')
    plt.show()

    full_m = m
    full_b = hp.alm2map(hp.map2alm(full_m, lmax=lmax)[2], nside=nside)
    cut_b = hp.alm2map(hp.map2alm(full_m, lmax=lmax)[2], nside=nside) * apo_mask
    res_b = cln_b - cut_b
    hp.orthview(hp.ma(res_b, badval=0), rot=[100,50,0],  half_sky=True, title='residual', min=-0.6, max=0.6, cmap='jet', badcolor='white')
    plt.show()

    full_cl = hp.anafast(full_b, lmax=lmax)
    cut_cl = hp.anafast(cut_b, lmax=lmax)
    cln_cl = hp.anafast(cln_b, lmax=lmax)
    crt_cl = hp.anafast(crt_b, lmax=lmax)
    # res_apo = hp.alm2map(hp.almxfl(hp.map2alm(res_b, lmax=lmax), 1/bl), nside=nside) * apo_mask
    res_cl = hp.anafast(res_b, lmax=lmax)
    l = np.arange(lmax+1)
    mask_cl = hp.anafast(apo_mask, lmax=lmax)

    plt.semilogy(l, l*(l+1)*full_cl/(2*np.pi))
    plt.semilogy(l, l*(l+1)*cut_cl/(2*np.pi)/apo_fsky)
    plt.semilogy(l, l*(l+1)*crt_cl/(2*np.pi)/apo_fsky)
    plt.semilogy(l, l*(l+1)*cln_cl/(2*np.pi)/apo_fsky)
    plt.semilogy(l, l*(l+1)*res_cl/(2*np.pi)/apo_fsky)


    plt.show()

def check_map():
    m = np.load('../../FGSim/FG/85.npy')
    lmax=1500
    nside=512
    l = np.arange(lmax+1)

    cl = hp.anafast(m, lmax=lmax)
    plt.loglog(l*(l+1)*cl[2]/(2*np.pi))
    plt.show()


def pseudo_Cl_alg():
    import pymaster as nmt
    nside=512
    lmax=500
    m = np.load('../../FGSim/CMB/270.npy')
    bl = hp.gauss_beam(np.deg2rad(9)/60, lmax=500, pol=True)

    # bl_curl = np.load('../smooth/BL/bl_std_curl.npy')
    # bl_temp = np.load('../smooth/BL/bl_std_temp.npy')
    # bl_grad = np.load('../smooth/BL/bl_std_grad.npy')

    alms = hp.map2alm(m, lmax=lmax)
    almT, almE, almB = [alm for alm in alms]
    almT = hp.almxfl(almT, 1/bl[:,0])
    almE = hp.almxfl(almE, 1/bl[:,1])
    almB = hp.almxfl(almB, 1/bl[:,2])
    m = hp.alm2map([almT, almE, almB], nside=nside)

    bin_mask = np.load('../mask/north/BINMASKG.npy')
    apo_mask = np.load('../mask/north/APOMASKC1_2.npy')

    cut_b = hp.alm2map(hp.map2alm(m, lmax=lmax)[2], nside=nside) * bin_mask

    obj = EBLeakageCorrection(m, lmax, nside, mask=bin_mask, post_mask=bin_mask, method='itercrtqu')
    crt_b, tmp_b, cln_b = obj.run_eblc()

    lkd_b = crt_b - cut_b
    res_b = cln_b - cut_b
    # lkd_b = hp.smoothing(lkd_b, fwhm=np.deg2rad(70)/60) * bin_mask

    # hp.orthview(lkd_b, rot=[100,50,0], half_sky=True, title='lkd_b', min=-0.7, max=0.7, cmap='jet')
    # hp.orthview(res_b, rot=[100,50,0], half_sky=True, title='res_b', min=-0.7, max=0.7, cmap='jet')
    # plt.show()

    f_lkd = nmt.NmtField(apo_mask, [lkd_b])

    b = nmt.NmtBin.from_lmax_linear(lmax, 40, is_Dell=True)
    dl_lkd = nmt.compute_full_master(f_lkd, f_lkd, b)

    f_res = nmt.NmtField(apo_mask, [res_b])
    dl_res = nmt.compute_full_master(f_res, f_res, b)

    cl = np.array([hp.anafast(m, lmax=lmax)[2]])
    print(f'{cl.shape = }')

    dl_true = b.bin_cell(cls_in=cl)

    ell_arr = b.get_effective_ells()

    plt.plot(ell_arr, dl_lkd[0],  label='lkd_b')
    plt.plot(ell_arr, dl_res[0],  label='res_b')
    plt.plot(ell_arr, dl_true[0],  label='true_b')
    plt.legend()
    plt.semilogy()
    # plt.xlim(0,300)
    # plt.ylim(1e-6,1e-2)
    plt.xlabel('$\\ell$', fontsize=16)
    plt.ylabel('$D_\\ell$', fontsize=16)
    plt.show()




if __name__ == '__main__':
    # check_eblc_with_smooth()
    # check_map()
    pseudo_Cl_alg()
