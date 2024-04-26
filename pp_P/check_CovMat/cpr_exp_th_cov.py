import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

th_T_cov = np.load('./Cov_T.npy')
th_QU_cov = np.load('./Cov_QU.npy')

exp_T_cov = np.load('./Exp_T_cov.npy')
exp_QU_cov = np.load('./Exp_QU_cov.npy')

print(f'{np.max(th_T_cov)=}')
print(f'{np.max(th_QU_cov)=}')

print(f'{np.max(exp_T_cov)=}')
print(f'{np.max(exp_QU_cov)=}')

def plot_T():
    plt.figure(1)
    cax = plt.imshow(np.abs(th_T_cov), cmap='viridis', norm=LogNorm(vmin=1e-2, vmax=1e3))
    plt.title('th')
    plt.colorbar(cax)
    plt.figure(2)
    cax = plt.imshow(np.abs(exp_T_cov), cmap='viridis', norm=LogNorm(vmin=1e-2, vmax=1e3))
    plt.title('exp')
    plt.colorbar(cax)
    plt.figure(3)
    cax = plt.imshow(np.abs((th_T_cov-exp_T_cov) / np.abs(th_T_cov)), cmap='viridis', norm=LogNorm(vmin=1e-3, vmax=1e0))
    plt.title('res')
    plt.colorbar(cax)
    plt.show()

def plot_QU():
    plt.figure(1)
    cax = plt.imshow(np.abs(th_QU_cov), cmap='viridis', norm=LogNorm( vmin=1e-3, vmax=1e1))
    plt.title('th')
    plt.colorbar(cax)
    plt.figure(2)
    cax = plt.imshow(np.abs(exp_QU_cov), cmap='viridis', norm=LogNorm(vmin=1e-3, vmax=1e1))
    plt.title('exp')
    plt.colorbar(cax)
    plt.figure(3)
    cax = plt.imshow(np.abs(th_QU_cov-exp_QU_cov)/np.abs(exp_QU_cov), cmap='viridis', norm=LogNorm(vmin=1e-3, vmax=1e1))
    plt.title('res')
    plt.colorbar(cax)
    plt.show()

plot_T()
plot_QU()

