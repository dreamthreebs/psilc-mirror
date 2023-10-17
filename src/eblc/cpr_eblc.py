from eblc import EBLeakageCorrection

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pymaster as nmt

plt.rcParams['lines.marker'] = 'o'
plt.rcParams['lines.markersize'] = 5

class CompareEBleakageCorrection:
    def __init__(self, m, lmax, nside, bin_mask, apo_mask, n_iter):
        ''' m should be some full sky IQU map '''
        self.m = m
        self.nside = nside
        self.lmax = lmax
        self.bin_mask = bin_mask
        self.apo_mask = apo_mask
        self.n_iter = n_iter

        self.cut_b = hp.alm2map(hp.map2alm(self.m, lmax=lmax)[2], nside=nside) * self.bin_mask

        self.b = nmt.NmtBin.from_lmax_linear(lmax, 40, is_Dell=True)
        self.ell_arr = self.b.get_effective_ells()

        self.bl_out = hp.gauss_beam(np.deg2rad(1), lmax=6400, pol=True)
        self.bl_out_binned = self.b.bin_cell(cls_in=np.array([bl_out[:,2]]))[0]

        cl = np.array([hp.anafast(self.m, lmax=lmax)[2]])
        print(f'{cl.shape = }')
        self.dl_true = self.b.bin_cell(cls_in=cl)[0]
        clr001 = np.load('../ebleakage/cmbdata/cmbcl3.npy')
        print(f'{clr001.shape=}')
        self.dlr001 = self.b.bin_cell(cls_in=np.array([clr001[:lmax+1,2]]))[0]


    def do_eblc(self, method):
        obj = EBLeakageCorrection(self.m, lmax=self.lmax, nside=self.nside, mask=self.bin_mask, post_mask=self.bin_mask, method=method, n_iter=self.n_iter)
        crt_b, tmp_b, cln_b = obj.run_eblc()

        lkg_b = crt_b - self.cut_b
        res_b = cln_b - self.cut_b
        return lkg_b, res_b

    def calc_dl_from_scalar_map(self, scalar_map):
        scalar_field = nmt.NmtField(self.apo_mask, [scalar_map], beam=self.bl_out[:,2])
        dl = nmt.compute_full_master(scalar_field, scalar_field, self.b)
        return dl[0]

    def change_beam(self, m):
        alm = hp.map2alm(m, lmax=lmax)
        print(f'{alm.shape=}')
        return hp.alm2map(hp.almxfl(alm, 1/self.bl_out[:,2]), nside=nside) * bin_mask

    def cpr_eblc(self, vmin=-0.7, vmax=0.7, cmap='jet'):
        lkg_b, res_b = self.do_eblc(method='fullqufitqu')
        print(f'{lkg_b.shape=}')
        dl_lkg = self.calc_dl_from_scalar_map(lkg_b)
        dl_res = self.calc_dl_from_scalar_map(res_b)
        hp.orthview(hp.ma(lkg_b, badval=0), rot=[100,50,0], half_sky=True, title='lkg_b fullqu', min=vmin, max=vmax, cmap=cmap, badcolor='white')
        hp.orthview(hp.ma(res_b, badval=0), rot=[100,50,0], half_sky=True, title='lkg_b fullqu', min=vmin, max=vmax, cmap=cmap, badcolor='white')
        plt.show()

        _, res_b1 = self.do_eblc(method='fullqufitb')
        hp.orthview(hp.ma(res_b1, badval=0), rot=[100,50,0], half_sky=True, title='res_b fullqufitb', min=vmin, max=vmax, cmap=cmap, badcolor='white')
        plt.show()

        dl_res1 = self.calc_dl_from_scalar_map(res_b1)

        # lkg_b2, res_b2 = self.do_eblc(method='cutqufitqu')
        # hp.orthview(hp.ma(lkg_b2, badval=0), rot=[100,50,0], half_sky=True, title='lkg_b cutqu', min=vmin, max=vmax, cmap=cmap, badcolor='white')
        # hp.orthview(hp.ma(res_b2, badval=0), rot=[100,50,0], half_sky=True, title='res_b fullqufitqu', min=vmin, max=vmax, cmap=cmap, badcolor='white')
        # plt.show()

        # dl_lkg2 = self.calc_dl_from_scalar_map(lkg_b2)
        # dl_res2 = self.calc_dl_from_scalar_map(res_b2)

        # _, res_b3 = self.do_eblc(method='cutqufitb')
        # dl_res3 = self.calc_dl_from_scalar_map(res_b3)

        # lkg_b4, res_b4 = self.do_eblc(method='zzr')

        # hp.orthview(hp.ma(lkg_b4, badval=0), rot=[100,50,0], half_sky=True,sub=(121), title='lkg_b zzr', min=vmin, max=vmax, cmap=cmap, badcolor='white')
        # hp.orthview(hp.ma(res_b4, badval=0), rot=[100,50,0], half_sky=True,sub=(122), title='res_b zzr', min=vmin, max=vmax, cmap=cmap, badcolor='white')
        # plt.show()

        # dl_lkg4 = self.calc_dl_from_scalar_map(lkg_b4)
        # dl_res4 = self.calc_dl_from_scalar_map(res_b4)

        # plt.plot(self.ell_arr, self.dl_true,  label='true_b only lensing', color='black')

        plt.plot(self.ell_arr, self.dlr001,  label='true_b r=0.01', color='black', linestyle=':')
        plt.plot(self.ell_arr, dl_lkg,  label='lkg_b fullqu')
        plt.plot(self.ell_arr, dl_res,  label='res_b fullqufitqu')
        plt.plot(self.ell_arr, dl_res1,  label='res_b fullqufitb')

        # plt.plot(self.ell_arr, dl_lkg2,  label='lkg_b cutqu')
        # plt.plot(self.ell_arr, dl_res2,  label='res_b cutqufitqu')
        # plt.plot(self.ell_arr, dl_res3,  label='res_b cutqufitb')

        # plt.plot(self.ell_arr, dl_lkg4,  label='lkg_b zzr')
        # plt.plot(self.ell_arr, dl_res4,  label='res_b zzr')


        plt.legend()
        plt.semilogy()
        # plt.xlim(0,300)
        plt.ylim(1e-6,1e-1)
        plt.xlabel('$\\ell$', fontsize=16)
        plt.ylabel('$D_\\ell$', fontsize=16)
        plt.show()

    def cpr_iter(self, vmin=-0.7, vmax=0.7, cmap='jet'):

        lkg_b, res_b = self.do_eblc(method='iterclncutqufitqu')
        print(f'{lkg_b.shape=}')
        dl_lkg = self.calc_dl_from_scalar_map(lkg_b)
        dl_res = self.calc_dl_from_scalar_map(res_b)

        # hp.orthview(hp.ma(lkg_b, badval=0), rot=[100,50,0], half_sky=True, title='lkg_b fullqu', min=vmin, max=vmax, cmap=cmap, badcolor='white')
        # plt.show()

        # hp.orthview(hp.ma(res_b, badval=0), rot=[100,50,0], half_sky=True, title='res_b fullqufitb', min=vmin, max=vmax, cmap=cmap, badcolor='white')
        # plt.show()

        # # plt.plot(self.ell_arr, self.dl_true,  label='true_b only lensing', color='black')
        # plt.plot(self.ell_arr, self.dlr001,  label='true_b r=0.01', color='black', linestyle=':')
        # plt.plot(self.ell_arr, dl_lkg,  label='lkg_b iter')
        # plt.plot(self.ell_arr, dl_res,  label='res_b iter')

        self.dl_lkg = dl_lkg
        self.dl_res = dl_res

        # plt.legend()
        # plt.semilogy()
        # # plt.xlim(0,300)
        plt.ylim(1e-6,1e-1)
        # plt.xlabel('$\\ell$', fontsize=16)
        # plt.ylabel('$D_\\ell$', fontsize=16)
        # plt.show()


if __name__ == '__main__':
    nside=2048
    lmax=500
    m = np.load('../smooth/TEST_PART/noPS_northNOI2048full/CMB/145.npy')
    bl = hp.gauss_beam(np.deg2rad(9)/60, lmax=500, pol=True)
    bl_out = hp.gauss_beam(np.deg2rad(1), lmax=500, pol=True)

    # bl_curl = np.load('../smooth/BL/bl_std_curl.npy')
    # bl_temp = np.load('../smooth/BL/bl_std_temp.npy')
    # bl_grad = np.load('../smooth/BL/bl_std_grad.npy')

    # alms = hp.map2alm(m, lmax=lmax)
    # almT, almE, almB = [alm for alm in alms]
    # almT = hp.almxfl(almT, 1/bl[:,0])
    # almE = hp.almxfl(almE, 1/bl[:,1])
    # almB = hp.almxfl(almB, 1/bl[:,2])
    # m = hp.alm2map([almT, almE, almB], nside=nside)

    alms = hp.map2alm(m, lmax=lmax)
    almT, almE, almB = [alm for alm in alms]
    almT = hp.almxfl(almT, bl_out[:,0]/bl[:,0])
    almE = hp.almxfl(almE, bl_out[:,1]/bl[:,1])
    almB = hp.almxfl(almB, bl_out[:,2]/bl[:,2])
    m = hp.alm2map([almT, almE, almB], nside=nside)


    bin_mask = np.load('../mask/north/BINMASKG2048.npy')
    apo_mask = np.load('../mask/north/APOMASK2048C1_2.npy')

    obj = CompareEBleakageCorrection(m, lmax, nside, bin_mask, apo_mask, n_iter=0)

    obj.cpr_iter()
    ell_arr = obj.ell_arr
    dlr001 = obj.dlr001

    lkg_iter0 = obj.dl_lkg
    res_iter0 = obj.dl_res
    plt.plot(ell_arr, dlr001,  label='true_b r=0.01', color='black', linestyle=':')
    plt.plot(ell_arr, lkg_iter0,  label='lkg_b')
    plt.plot(ell_arr, res_iter0,  label='res_b iter0')

    obj = CompareEBleakageCorrection(m, lmax, nside, bin_mask, apo_mask, n_iter=1)
    obj.cpr_iter()

    res_iter1 = obj.dl_res

    plt.plot(ell_arr, res_iter1,  label='res_b iter1')


    obj = CompareEBleakageCorrection(m, lmax, nside, bin_mask, apo_mask, n_iter=2)
    obj.cpr_iter()

    res_iter2 = obj.dl_res

    plt.plot(ell_arr, res_iter2,  label='res_b iter2')


    obj = CompareEBleakageCorrection(m, lmax, nside, bin_mask, apo_mask, n_iter=3)
    obj.cpr_iter()

    res_iter3 = obj.dl_res

    plt.plot(ell_arr, res_iter3,  label='res_b iter3')

    obj = CompareEBleakageCorrection(m, lmax, nside, bin_mask, apo_mask, n_iter=4)
    obj.cpr_iter()

    res_iter4 = obj.dl_res

    plt.plot(ell_arr, res_iter4,  label='res_b iter4')

    obj = CompareEBleakageCorrection(m, lmax, nside, bin_mask, apo_mask, n_iter=9)
    obj.cpr_iter()

    # res_iter9 = obj.dl_res
    # plt.plot(ell_arr, res_iter9,  label='res_b iter9')

    # # obj = CompareEBleakageCorrection(m, lmax, nside, bin_mask, apo_mask, n_iter=15)
    # # obj.cpr_iter()

    # # res_iter15 = obj.dl_res
    # # plt.plot(ell_arr, res_iter15,  label='res_b iter15')

    # # obj = CompareEBleakageCorrection(m, lmax, nside, bin_mask, apo_mask, n_iter=50)
    # # obj.cpr_iter()

    # # res_iter50 = obj.dl_res
    # # plt.plot(ell_arr, res_iter50,  label='res_b iter50')

    # # obj = CompareEBleakageCorrection(m, lmax, nside, bin_mask, apo_mask, n_iter=100)
    # # obj.cpr_iter()

    # # res_iter100 = obj.dl_res
    # # plt.plot(ell_arr, res_iter100,  label='res_b iter100')

    # plt.legend()
    # plt.semilogy()
    # # plt.xlim(0,300)
    # # plt.ylim(1e-6,1e-1)
    # plt.xlabel('$\\ell$', fontsize=16)
    # plt.ylabel('$D_\\ell$', fontsize=16)
    # plt.show()



    # obj.cpr_eblc()
