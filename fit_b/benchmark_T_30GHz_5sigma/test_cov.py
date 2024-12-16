import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from ctypes import *
from pathlib import Path

from config import lmax, beam

flux_idx=0

class CovCalculator:
    @staticmethod
    def is_pos_def(M):
        eigvals = np.linalg.eigvals(M)
        plt.plot(eigvals)
        plt.show()
        print(np.min(eigvals))
        return np.all(eigvals>-1e-5)
    @staticmethod
    def check_symmetric(m, rtol=1e-05, atol=1e-05):
        return np.allclose(m, m.T, rtol=rtol, atol=atol)

    def __init__(self, nside, lmin, lmax, Cl_TT, Cl_EE, Cl_BB, Cl_TE, pixind, calc_opt='scalar', out_pol_opt=None):
        # the input Cl should range from 0 to >lmax
        self.nside = nside
        self.lmin = lmin
        self.lmax = lmax
        self.Cl_TT = Cl_TT[lmin:lmax+1].copy()

        if calc_opt == 'scalar':
            pass
        elif calc_opt == 'polarization':
            self.Cl_EE = Cl_EE[lmin:lmax+1].copy()
            self.Cl_BB = Cl_BB[lmin:lmax+1].copy()
            self.Cl_TE = Cl_TE[lmin:lmax+1].copy()
        else:
            raise ValueError('calc_opt should be scalar or polarization!')

        if np.size(self.Cl_TT) < (lmax+1-lmin):
            raise ValueError('input Cl size < l range')

        self.pixind = pixind
        self.calc_opt = calc_opt
        self.out_pol_opt = out_pol_opt

        self.l = np.arange(lmin, lmax+1).astype(np.float64)

    def Calc_CovMat(self):
        pixind = self.pixind
        nside = self.nside
        l = self.l
        nl = len(l) # number of ells
        print(f'number of l = {nl}')

        nCl = np.zeros((nl,))
        print(f'{nCl.shape=}')

        if self.calc_opt=='scalar':
            Cls = np.array([self.Cl_TT, nCl, nCl, nCl, nCl]) # TT,EE,BB,TE,TB
        elif self.calc_opt=='polarization':
            Cls = np.array([self.Cl_TT, self.Cl_EE, self.Cl_BB, self.Cl_TE, nCl])

        npix = len(pixind)
        vecst = hp.pix2vec(nside, pixind)
        vecs = np.array(vecst).T
        covmat = np.zeros((3*npix, 3*npix), dtype=np.float64)

        # use the c package to calculate the Covmat
        lib = cdll.LoadLibrary('../CovMat.so')
        CovMat = lib.CovMat
        CovMat(c_void_p(vecs.ctypes.data), c_void_p(l.ctypes.data), c_void_p(Cls.ctypes.data), c_void_p(covmat.ctypes.data), c_int(npix), c_int(nl))
        # covert back to 2d
        covmat = covmat.reshape((3*npix, 3*npix))
        return covmat

    def ChooseMat(self, M):
        if self.out_pol_opt is None:
            MP = M[0:len(M):3, 0:len(M):3] # just like TT
            print(f'{MP.shape=}')
        if self.out_pol_opt=='QQ':
            MP = M[1:len(M)+1:3, 1:len(M)+1:3]
        if self.out_pol_opt=='UU':
            MP = M[2:len(M)+2:3, 2:len(M)+2:3]
        if self.out_pol_opt=='QU':
            QQ = M[1:len(M)+1:3, 1:len(M)+1:3]
            print(f'{QQ.shape=}')
            QU = M[1:len(M)+1:3, 2:len(M)+2:3]
            UQ = M[2:len(M)+2:3, 1:len(M)+1:3]
            UU = M[2:len(M)+2:3, 2:len(M)+2:3]
            MP = np.block([[QQ,QU],[UQ,UU]])
            print(f'{MP.shape=}')
        return MP

    def run_calc_cov(self):
        M = self.Calc_CovMat()
        MP = self.ChooseMat(M)
        return MP

# tests
def test_cmb_cl(beam, lmax):
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=10000, pol=True)
    print(f'{bl[0:10,0]=}')
    print(f'{bl[0:10,1]=}')
    print(f'{bl[0:10,2]=}')
    print(f'{bl[0:10,3]=}')
    # cl = np.load('../../src/cmbsim/cmbdata/cmbcl.npy')
    cl = np.load('../../src/cmbsim/cmbdata/cmbcl_8k.npy')
    print(f'{cl.shape=}')

    Cl_TT = cl[0:lmax+1,0] * bl[0:lmax+1,0]**2
    Cl_EE = cl[0:lmax+1,1] * bl[0:lmax+1,1]**2
    Cl_BB = cl[0:lmax+1,2] * bl[0:lmax+1,2]**2
    Cl_TE = cl[0:lmax+1,3] * bl[0:lmax+1,3]**2
    return Cl_TT, Cl_EE, Cl_BB, Cl_TE

def test_fg_cl():
    cl_fg = np.load('./data/debeam_full_b/cl_fg.npy')
    print(f'{cl_fg.shape=}')
    Cl_TT = cl_fg[0]
    Cl_EE = cl_fg[1]
    Cl_BB = cl_fg[2]
    Cl_TE = np.zeros_like(Cl_TT)
    return Cl_TT, Cl_EE, Cl_BB, Cl_TE

def test_noise_cl(lmax):
    map_depth = 1.9
    Cl_NN = (map_depth*np.ones(shape=(lmax+1,)))**2 / 3437.748**2 # 3437.748 is the factor from sr to arcmin**2
    print(f'{Cl_NN.shape=}')
    return Cl_NN

def main_cn():
    nside = 2048
    lmin = 2
    # lmax = 3 * nside - 1
    lmax = 1000
    flux_idx=0
    beam = 67
    pixind = np.load(f'./pix_idx_qu/{flux_idx}.npy')

    ## test for cmb cov

    Cl_TT,Cl_EE,Cl_BB,Cl_TE = test_cmb_cl(beam=beam, lmax=1000)

    l = np.arange(lmax + 1)
    plt.loglog(l, l*(l+1)*Cl_BB/(2*np.pi))
    plt.loglog(l, l*(l+1)*Cl_TT/(2*np.pi))
    plt.loglog(l, l*(l+1)*Cl_EE/(2*np.pi))
    plt.show()

    # obj = CovCalculator(nside=nside, lmin=2, lmax=lmax, Cl_TT=Cl_TT, Cl_EE=Cl_EE, Cl_BB=Cl_BB, Cl_TE=Cl_TE, pixind=pixind, calc_opt='polarization', out_pol_opt='QU')
    # MP = obj.run_calc_cov()
    # path_test_class_cov = Path('./cmb_qu_cov')
    # path_test_class_cov.mkdir(exist_ok=True, parents=True)
    # # np.save(f'./test_class_cov/cmb.npy', MP)
    # np.save(path_test_class_cov / Path(f'{flux_idx}.npy'), MP)

    ## test for noise cov

    # Cl_NN = test_noise_cl(lmax=3*nside-1)
    # obj = CovCalculator(nside=nside, lmin=2, lmax=3*nside-1, Cl_TT=Cl_NN, Cl_EE=None, Cl_BB=None, Cl_TE=None, pixind=pixind)
    # MP = obj.run_calc_cov()
    # path_test_class_cov = Path('./test_class_cov')
    # path_test_class_cov.mkdir(exist_ok=True, parents=True)
    # np.save(f'./test_class_cov/noise.npy', MP)

def main_fg():
    nside = 2048
    lmin = 2
    # lmax = 3 * nside - 1
    lmax = 600
    flux_idx=0
    beam = 67
    pixind = np.load(f'./pix_idx_qu/{flux_idx}.npy')

    ## test for cmb cov

    Cl_TT,Cl_EE,Cl_BB,Cl_TE = test_fg_cl()

    # l = np.arange(lmax + 1)
    # plt.loglog(l, l*(l+1)*Cl_BB/(2*np.pi))
    # plt.loglog(l, l*(l+1)*Cl_TT/(2*np.pi))
    # plt.loglog(l, l*(l+1)*Cl_EE/(2*np.pi))
    # plt.show()

    obj = CovCalculator(nside=nside, lmin=2, lmax=lmax, Cl_TT=Cl_TT, Cl_EE=Cl_EE, Cl_BB=Cl_BB, Cl_TE=Cl_TE, pixind=pixind, calc_opt='polarization', out_pol_opt='QU')
    MP = obj.run_calc_cov()
    path_test_class_cov = Path('./fg_qu_cov')
    path_test_class_cov.mkdir(exist_ok=True, parents=True)
    # np.save(f'./test_class_cov/cmb.npy', MP)
    np.save(Path(path_test_class_cov / f'{flux_idx}.npy'), MP)

    ## test for noise cov

    # Cl_NN = test_noise_cl(lmax=3*nside-1)
    # obj = CovCalculator(nside=nside, lmin=2, lmax=3*nside-1, Cl_TT=Cl_NN, Cl_EE=None, Cl_BB=None, Cl_TE=None, pixind=pixind)
    # MP = obj.run_calc_cov()
    # path_test_class_cov = Path('./test_class_cov')
    # path_test_class_cov.mkdir(exist_ok=True, parents=True)
    # np.save(f'./test_class_cov/noise.npy', MP)

def main_cf():
    nside = 1024
    lmin = 2
    # lmax = 3 * nside - 1
    pixind = np.load(f'./pix_idx_qu/{flux_idx}.npy')

    ## test for cmb cov

    Cl_TT_FG,Cl_EE_FG,Cl_BB_FG,Cl_TE_FG = test_fg_cl()
    Cl_TT_CMB,Cl_EE_CMB,Cl_BB_CMB,Cl_TE_CMB = test_cmb_cl(beam=beam, lmax=lmax)

    Cl_TT = Cl_TT_CMB + Cl_TT_FG
    Cl_EE = Cl_EE_CMB + Cl_EE_FG
    Cl_BB = Cl_BB_CMB + Cl_BB_FG
    Cl_TE = Cl_TE_CMB + Cl_TE_FG

    l = np.arange(lmax + 1)
    plt.loglog(l, l*(l+1)*Cl_BB/(2*np.pi), label='BB')
    plt.loglog(l, l*(l+1)*Cl_TT/(2*np.pi), label='TT')
    plt.loglog(l, l*(l+1)*Cl_EE/(2*np.pi), label='EE')
    plt.legend()
    plt.show()

    obj = CovCalculator(nside=nside, lmin=2, lmax=lmax, Cl_TT=Cl_TT, Cl_EE=Cl_EE, Cl_BB=Cl_BB, Cl_TE=Cl_TE, pixind=pixind, calc_opt='polarization', out_pol_opt='QU')
    MP = obj.run_calc_cov()

    path_test_class_cov = Path('./cmb_qu_cov')
    path_test_class_cov.mkdir(exist_ok=True, parents=True)
    # np.save(f'./test_class_cov/cmb.npy', MP)
    np.save(Path(path_test_class_cov / f'{flux_idx}.npy'), MP)



if __name__=='__main__':
    main_cf()




