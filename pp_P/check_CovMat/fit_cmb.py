import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from iminuit import Minuit
from pathlib import Path

m = np.load('./data/33.npy')[0]
nside = 8
npix = hp.nside2npix(nside)
ipix_fit = np.arange(npix)

cov = np.load('./Cov_T.npy')
eigenval, eigenvec = np.linalg.eigh(cov)
print(f'{eigenval=}')
eigenval[eigenval < 0] = 0
reconstructed_cov = np.dot(eigenvec * eigenval, eigenvec.T)
print(f'{np.max(np.abs(reconstructed_cov-cov))=}')
print(f'{reconstructed_cov=}')
cov  = reconstructed_cov + 0.1 * np.eye(cov.shape[0])


inv_cov = np.linalg.inv(cov)
I = cov @ inv_cov
print(f'{I=}')

def lsq(const):
    y_data = m
    y_diff = y_data - const
    z = y_diff @ inv_cov @ y_diff
    return z

obj_minuit = Minuit(lsq, const=0)
obj_minuit.limits = [(-1000,1000)]
print(obj_minuit.migrad())
chi2dof = obj_minuit.fval / npix
str_chi2 = f"𝜒²/ndof = {obj_minuit.fval:.2f} / {npix} = {chi2dof}"
print(str_chi2)
