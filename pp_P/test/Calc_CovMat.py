import numpy as np
import healpy as hp
# import pyslalib as sla
from scipy import special
import matplotlib.pyplot as plt
from ctypes import *
# from util import Mask_Operation

def Calc_CovMat(l_range, nside, opt='E', mskopt=True):
    l = l_range.astype(np.float64)
    nl = len(l)
    print(f'{nl=}')


    lmax = 2000
    beam = 9
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam)/60, lmax=10000)
    bl = bl[2:nl+2]
    Cl = np.load('../../src/cmbsim/cmbdata/cmbcl.npy')[2:nl+2,0] * bl**2

    # bl = hp.gauss_beam(0.25*np.pi/180, lmax=10000)
    # bl = bl[2:nl+2]
    # Cl = np.pi*2/(l*(l+1))*(bl**2)
    #Cl = np.pi*2./(l**2)*(bl**2)
    #Cl = np.pi*2/(l*(l+1))
    #Cl = 1./((l+2)*(l+1)*l*(l-1))*(bl**2)
    nCl = np.zeros_like(Cl)
    
    if opt=='E':
    	Cls = np.array([Cl, Cl, nCl, Cl, nCl])
    elif opt=='B':
    	Cls = np.array([Cl, nCl, Cl, nCl, Cl])
    elif opt=='T':
        Cls = np.array([Cl, nCl, nCl, nCl, nCl])
    if mskopt ==True:
    	pixind = Mask_Operation(nside)
    	npix = len(pixind)
    else:
    	# npix = hp.nside2npix(nside)
    	# pixind = np.arange(npix)
        pixind = np.load('./ipix_fit.npy')

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

def ChooseMat(M, opt='Pol'):
	ind0 = np.arange(0, len(M), 3)
	if opt=='Pol':
		indtot = np.arange(len(M))
		ind = np.setdiff1d(indtot, ind0)
	if opt=='T':
		ind = ind0
	xind, yind = np.meshgrid(ind, ind)
	MP = M[xind, yind]
	return MP

if __name__=='__main__':
    l_range = np.arange(2,2000)
    nside = 2048
    mskopt = False
    CovMat = Calc_CovMat(l_range, nside, opt='T', mskopt=mskopt)
    print(f'{CovMat.shape=}')
    CovMatT = ChooseMat(CovMat, opt='T')
    CovMatPol = ChooseMat(CovMat, opt='Pol')
    print(f'{CovMatT.shape=}')
    print(f'{CovMatPol.shape=}')
    np.save('Cov_T.npy', CovMatT)
    
    print(CovMat)
    print(CovMatT)
    print(check_symmetric(CovMatT))
    print(is_pos_def(CovMatT))
    #eigvals = np.diag(CovMatT)
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #cax = ax.matshow(diff)
    ##cax = ax.matshow(CovMatT, vmin=-1, vmax=1)
    #fig.colorbar(cax)
    #plt.show()
