import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import pickle
import os

from iminuit import Minuit
from iminuit.cost import LeastSquares
from numpy.polynomial.legendre import Legendre
from scipy.interpolate import CubicSpline


n_samples = 1
cov = np.load('../../../test/fit_ps/cmb_cov_data/lmax500rf0.8.npy')

epsilon = 1e-4
cov = cov + epsilon * np.eye(cov.shape[0])

print(f'{cov.shape[0]=}')
n_dim = cov.shape[1]

def make_positive_semidefinite(matrix):
    # 特征值分解
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)

    # 将负特征值置为0
    eigenvalues[eigenvalues < 0] = 0

    # 重建矩阵
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

# 示例
# cov = make_positive_semidefinite(cov)



# A = np.random.rand(n_dim, n_dim)
# cov1 = np.dot(A, A.transpose())
# cov = cov1[:500,:500]
# print(f'{cov.shape=}')

mean_vec = np.zeros(n_dim)

data = np.random.multivariate_normal(mean_vec, cov, n_samples)
print(f'{data =}')
print(f'{data.shape =}')
inv_cov = np.linalg.inv(cov)
ndof = data.shape[1]

def model():
    return 0

def lsq_1_params(const):

    y_model = model()
    y_data = data[0]
    y_diff = y_data - y_model
    print(f'{y_diff=}')

    z = (y_diff) @ inv_cov @ (y_diff)
    return z

obj_minuit = Minuit(lsq_1_params, const=0.0)
obj_minuit.limits = [(-1000,1000),]
print(obj_minuit.migrad())
print(obj_minuit.hesse())
chi2dof = obj_minuit.fval / ndof
str_chi2 = f"𝜒²/ndof = {obj_minuit.fval:.2f} / {ndof} = {chi2dof}"
print(str_chi2)






