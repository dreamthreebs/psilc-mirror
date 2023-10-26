from .eblc_base import EBLeakageCorrection

import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import glob
import pymaster as nmt

class EBLeakageCorrectionPipeline():
    def __init__(self, m_list, lmax, nside, bin_mask, apo_mask, method, save_path, n_iter=1):
        ''' m in IQU '''
        self.m_list = m_list
        self.lmax = lmax
        self.nside = nside
        self.bin_mask = bin_mask
        self.apo_mask = apo_mask
        self.n_iter = n_iter
        self.method = method
        # self.df = pd.read_csv('../../FGSim/FreqBand')
        self.b = nmt.NmtBin.from_lmax_linear(lmax, 40, is_Dell=True)
        self.ell_arr = self.b.get_effective_ells()
        self.save_path = save_path

    def do_eblc(self, m, method, n_iter=3):
        obj = EBLeakageCorrection(m, lmax=self.lmax, nside=self.nside, mask=self.bin_mask, post_mask=self.bin_mask, method=method, check_res=False, n_iter=n_iter)
        self.crt_b, self.tmp_b, self.cln_b = obj.run_eblc()

    def do_eblc_for_check(self, m, method, n_iter=3):
        obj = EBLeakageCorrection(m, lmax=self.lmax, nside=self.nside, mask=self.bin_mask, post_mask=self.bin_mask, method=method, check_res=True, n_iter=n_iter)
        self.crt_b, self.tmp_b, self.cln_b = obj.run_eblc()

        self.cut_b = hp.alm2map(hp.map2alm(self.true_map, lmax=lmax)[2], nside=nside) * self.bin_mask
        self.lkg_b = self.crt_b - self.cut_b
        self.res_b = self.cln_b - self.cut_b

    def io_pipeline_for_check(self, method, lmax, nside, n_iter):
        l = np.arange(lmax+1)
        for index, maps in enumerate(self.m_list):
            freq = int(Path(maps).stem)
            print(f'frequency:{freq}')
            self.true_map = np.load(maps)
            self.do_eblc_for_check(self.true_map, method=method, n_iter=n_iter)

            bl = hp.gauss_beam(np.deg2rad(self.df.at[index, 'beam'])/60, lmax=6400, pol=True)[:,2]
            dl_cut = self.calc_dl_from_scalar_map(self.cut_b, bl=bl)
            dl_lkg = self.calc_dl_from_scalar_map(self.lkg_b, bl=bl)
            dl_res = self.calc_dl_from_scalar_map(self.res_b, bl=bl)
            plt.plot(self.ell_arr, dl_cut,  label=f'cut_b at freq:{freq}')
            plt.plot(self.ell_arr, dl_lkg,  label=f'lkg_b at freq:{freq}')
            plt.plot(self.ell_arr, dl_res,  label=f'res_b at freq:{freq}')

            plt.legend()
            plt.semilogy()
            # plt.xlim(0,300)
            # plt.ylim(1e-6,1e-1)
            plt.xlabel('$\\ell$', fontsize=16)
            plt.ylabel('$D_\\ell$', fontsize=16)
            plt.show()

            # self.cut_b = hp.alm2map(hp.map2alm(self.true_map, lmax=lmax)[2], nside=nside) * self.apo_mask
            # self.std_cl = hp.anafast(self.cut_b, lmax=lmax)
            # self.crt_cl = hp.anafast(self.crt_b, lmax=lmax)
            # self.cln_cl = hp.anafast(self.cln_b, lmax=lmax)
            # plt.semilogy(l*(l+1)*self.std_cl/(2*np.pi), label='std_cl')
            # plt.semilogy(l*(l+1)*self.crt_cl/(2*np.pi), label='crt_cl')
            # plt.semilogy(l*(l+1)*self.cln_cl/(2*np.pi), label='cln_cl')
            # plt.show()

    def io_pipeline(self, method, lmax, nside):
        l = np.arange(lmax+1)
        for index, maps in enumerate(self.m_list):
            freq = int(Path(maps).stem)
            print(f'frequency:{freq}')
            self.true_map = np.load(maps)
            self.do_eblc(self.true_map, method=method)
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            np.save(f'{self.save_path}/{freq}.npy', self.cln_b)

    def calc_dl_from_scalar_map(self, scalar_map, bl):
        scalar_field = nmt.NmtField(self.apo_mask, [scalar_map], beam=bl)
        dl = nmt.compute_full_master(scalar_field, scalar_field, self.b)
        return dl[0]

    def class_main(self):
        # self.io_pipeline_for_check(method='itercrtqu', lmax=self.lmax, nside=self.nside, n_iter=self.n_iter)
        self.io_pipeline(method=self.method, lmax=self.lmax, nside=self.nside)


if __name__ == '__main__':
    lmax = 500
    nside = 512
    bin_mask = np.load('../mask/north/BINMASKG.npy')
    apo_mask = np.load('../mask/north/APOMASKC1_2.npy')
    cmb_all = glob.glob('../sim/NSIDE512/noPS/SIM/*.npy')
    sorted_cmb = sorted(cmb_all, key=lambda x: int(Path(x).stem))
    print(f'{sorted_cmb=}')

    # eblc_obj = EBLeakageCorrectionPipeline(m_list=sorted_cmb, lmax=lmax, nside=nside, bin_mask=bin_mask, apo_mask=apo_mask, n_iter=5, save_path=None)
    # eblc_obj.class_main()

    # save_path = './eblc_data/sim'
    save_path = None
    eblc_obj = EBLeakageCorrectionPipeline(m_list=sorted_cmb, lmax=lmax, nside=nside, bin_mask=bin_mask, apo_mask=apo_mask, n_iter=3, save_path=save_path, method='cutqufitqu')
    eblc_obj.class_main()


