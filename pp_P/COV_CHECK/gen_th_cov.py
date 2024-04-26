import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from ctypes import *
from pathlib import Path

nside = 8192
lmax = 3 * 64 - 1
p1 = hp.ang2pix(nside=nside, theta=-1, phi=1, lonlat=True)
p2 = hp.ang2pix(nside=nside, theta=1, phi=1, lonlat=True)
p3 = hp.ang2pix(nside=nside, theta=1, phi=-1, lonlat=True)
p4 = hp.ang2pix(nside=nside, theta=-1, phi=-1, lonlat=True)
ipix_fit = np.array([p1,p2,p3,p4])
print(f'{ipix_fit=}')

def Calc_CovMat(l_range, nside, opt='E', mskopt=True):
    l = l_range.astype(np.float64)
    nl = len(l)
    print(f'{nl=}')

    Cl_TT = np.load('../../src/cmbsim/cmbdata/cmbcl.npy')[2:nl+2,0]
    Cl_EE = np.load('../../src/cmbsim/cmbdata/cmbcl.npy')[2:nl+2,1]
    Cl_BB = np.load('../../src/cmbsim/cmbdata/cmbcl.npy')[2:nl+2,2]
    Cl_TE = np.load('../../src/cmbsim/cmbdata/cmbcl.npy')[2:nl+2,3]

    # bl = hp.gauss_beam(0.25*np.pi/180, lmax=10000)
    # bl = bl[2:nl+2]
    # Cl = np.pi*2/(l*(l+1))*(bl**2)
    #Cl = np.pi*2./(l**2)*(bl**2)
    #Cl = np.pi*2/(l*(l+1))
    #Cl = 1./((l+2)*(l+1)*l*(l-1))*(bl**2)
    nCl = np.zeros_like(Cl_TT)

    if opt=='E':
    	Cls = np.array([Cl, Cl, nCl, Cl, nCl]) # TT,EE,BB,TE,TB
    elif opt=='B':
    	Cls = np.array([Cl, nCl, Cl, nCl, Cl])
    elif opt=='T':
        Cls = np.array([Cl, nCl, nCl, nCl, nCl])
    elif opt=='all':
        Cls = np.array([Cl_TT, Cl_EE, Cl_BB, Cl_TE, nCl])

    if mskopt ==True:
    	pixind = Mask_Operation(nside)
    	npix = len(pixind)
    else:
    	# npix = hp.nside2npix(nside)
    	# pixind = np.arange(npix)
        pixind = ipix_fit

    # print('npix=', npix)
    npix = len(pixind)
    vecst = hp.pix2vec(nside, pixind)
    vecs = np.array(vecst).T
    covmat = np.zeros((3*npix, 3*npix), dtype=np.float64)

    # use the c package to calculate the Covmat
    lib = cdll.LoadLibrary('./CovMat.so')
    #lib = cdll.LoadLibrary('./util/CovMat.so')
    CovMat = lib.CovMat
    CovMat(c_void_p(vecs.ctypes.data), c_void_p(l.ctypes.data), c_void_p(Cls.ctypes.data), c_void_p(covmat.ctypes.data), c_int(npix), c_int(nl))
    # covert back to 2d
    covmat = covmat.reshape((3*npix, 3*npix))
    return covmat

def is_pos_def(M):
	eigvals = np.linalg.eigvals(M)
	plt.plot(eigvals)
	plt.show()
	print(np.min(eigvals))
	return np.all(eigvals>-1e-5)

def check_symmetric(m, rtol=1e-05, atol=1e-05):
	return np.allclose(m, m.T, rtol=rtol, atol=atol)

def ChooseMat(M, opt):
    indi = np.arange(0, len(M), 3)
    indq = np.arange(1, len(M)+1, 3)
    indu = np.arange(2, len(M)+2, 3)
    print(f'{indi=}')
    print(f'{indq=}')
    print(f'{indu=}')
    print(f'{len(M)=}')

    if opt=='all':
        QQ = M[1:len(M)+1:3, 1:len(M)+1:3]
        print(f'{QQ.shape=}')
        QU = M[1:len(M)+1:3, 2:len(M)+2:3]
        # QU = np.zeros_like(QQ)
        UQ = M[2:len(M)+2:3, 1:len(M)+1:3]
        # UQ = np.zeros_like(QQ)
        UU = M[2:len(M)+2:3, 2:len(M)+2:3]
        MP = np.block([[QQ,QU],[UQ,UU]])
        # MP = combine_matrices([QQ, QU, UQ, UU])
    if opt=='QQ':
        MP = M[1:len(M)+1:3, 1:len(M)+1:3]
    if opt=='UU':
        MP = M[2:len(M)+2:3, 2:len(M)+2:3]

    return MP

if __name__=='__main__':
    l_range = np.arange(2,lmax)
    mskopt = False
    CovMat = Calc_CovMat(l_range, nside, opt='all', mskopt=mskopt)
    print(f'{len(CovMat)=}')
    print(f'{CovMat.shape=}')
    # CovMatT = ChooseMat(CovMat, opt='T')
    CovMatPol = ChooseMat(CovMat, opt='all')
    # print(f'{CovMatT.shape=}')
    print(f'{CovMatPol.shape=}')
    path_cov = Path(f'./{nside}/th')
    path_cov.mkdir(exist_ok=True, parents=True)

    np.save(path_cov / Path('cov_qu.npy'), CovMatPol)
    # np.save('Cov_U.npy', CovMatPol)

    print(CovMat)
    # np.set_printoptions(threshold=np.inf)
    # print(CovMatPol[196:392,0:196])

    # print(CovMatT)
    print(check_symmetric(CovMatPol))
    print(is_pos_def(CovMatPol))
    #eigvals = np.diag(CovMatT)
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #cax = ax.matshow(diff)
    ##cax = ax.matshow(CovMatT, vmin=-1, vmax=1)
    #fig.colorbar(cax)
    #plt.show()

