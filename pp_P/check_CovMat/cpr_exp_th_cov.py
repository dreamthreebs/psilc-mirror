import numpy as np
import matplotlib.pyplot as plt

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
    cax = plt.imshow(th_T_cov, cmap='viridis')
    plt.title('th')
    plt.colorbar(cax)
    plt.figure(2)
    cax = plt.imshow(exp_T_cov, cmap='viridis')
    plt.title('exp')
    plt.colorbar(cax)
    plt.figure(3)
    cax = plt.imshow(np.abs((th_T_cov-exp_T_cov) / th_T_cov), cmap='viridis')
    plt.title('res')
    plt.colorbar(cax)
    plt.show()

def plot_QU():
    plt.figure(1)
    cax = plt.imshow(th_QU_cov, cmap='viridis')
    plt.title('th')
    plt.colorbar(cax)
    plt.figure(2)
    cax = plt.imshow(exp_QU_cov, cmap='viridis')
    plt.title('exp')
    plt.colorbar(cax)
    plt.figure(3)
    cax = plt.imshow((th_QU_cov-exp_QU_cov), cmap='viridis')
    plt.title('res')
    plt.colorbar(cax)
    plt.show()

plot_T()
plot_QU()

