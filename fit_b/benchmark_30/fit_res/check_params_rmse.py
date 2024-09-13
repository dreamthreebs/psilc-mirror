import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

flux_idx = 0

rmse_q_cn_list = []
rmse_q_cfn_list = []
rmse_u_cn_list = []
rmse_u_cfn_list = []


for flux_idx in range(10):
    print(f'{flux_idx=}')
    P_true_value = np.load(f'./pcfn_params/fit_qu/idx_{flux_idx}/P_0.npy')
    phi_true_value = np.load(f'./pcfn_params/fit_qu/idx_{flux_idx}/phi_0.npy')
    
    q_true_value = P_true_value * np.cos(phi_true_value)
    u_true_value = P_true_value * np.sin(phi_true_value)

    print(f'{q_true_value=}')
    print(f'{u_true_value=}')
    
    
    q_cn_list = []
    u_cn_list = []
    q_cfn_list = []
    u_cfn_list = []
    
    for rlz_idx in range(200):
        fit_P_cn = np.load(f'./pcfn_params/fit_qu/idx_{flux_idx}/fit_P_{rlz_idx}.npy')
        fit_phi_cn = np.load(f'./pcfn_params/fit_qu/idx_{flux_idx}/fit_phi_{rlz_idx}.npy')
        fit_q_cn = fit_P_cn * np.cos(fit_phi_cn)
        fit_u_cn = fit_P_cn * np.sin(fit_phi_cn)
        q_cn_list.append(fit_q_cn)
        u_cn_list.append(fit_u_cn)
    
        fit_P_cfn = np.load(f'./pcfn_params/fit_qu_fg/idx_{flux_idx}/fit_P_{rlz_idx}.npy')
        fit_phi_cfn = np.load(f'./pcfn_params/fit_qu_fg/idx_{flux_idx}/fit_phi_{rlz_idx}.npy')
        fit_q_cfn = fit_P_cfn * np.cos(fit_phi_cfn)
        fit_u_cfn = fit_P_cfn * np.sin(fit_phi_cfn)
        q_cfn_list.append(fit_q_cfn)
        u_cfn_list.append(fit_u_cfn)
    
    q_cn_arr = np.array(q_cn_list)
    u_cn_arr = np.array(u_cn_list)
    
    q_cfn_arr = np.array(q_cfn_list)
    u_cfn_arr = np.array(u_cfn_list)

    print(f'{q_cn_arr=}')
    print(f'{u_cn_arr=}')
    print(f'{q_cfn_arr=}')
    print(f'{u_cfn_arr=}')
    
    
    nsim = len(q_cn_arr)
    
    rmse_q_cn = np.sqrt(np.sum((q_cn_arr - q_true_value)**2) / nsim)
    rmse_q_cfn = np.sqrt(np.sum((q_cfn_arr - q_true_value)**2) / nsim)
    
    
    rmse_u_cn = np.sqrt(np.sum((u_cn_arr - u_true_value)**2) / nsim)
    rmse_u_cfn = np.sqrt(np.sum((u_cfn_arr - u_true_value)**2) / nsim)
    print(f'{rmse_q_cn=}')
    print(f'{rmse_q_cfn=}')
    
    print(f'{rmse_u_cn=}')
    print(f'{rmse_u_cfn=}')

    rmse_q_cn_list.append(rmse_q_cn)
    rmse_q_cfn_list.append(rmse_q_cfn)
    rmse_u_cn_list.append(rmse_u_cn)
    rmse_u_cfn_list.append(rmse_u_cfn)

x = np.arange(10)

plt.plot(x, rmse_q_cn_list, label='Q cov(CMB+noise)')
plt.plot(x, rmse_q_cfn_list, label='Q cov(CMB+noise+diffuse fg)')
plt.legend()
plt.xlabel('flux_idx')
plt.ylabel('point source flux density RMSE')
# plt.show()

plt.savefig('/afs/ihep.ac.cn/users/w/wangyiming25/tmp/20230912/cov_q_rmse.png', dpi=300)
plt.show()

plt.plot(x, rmse_u_cn_list, label='U cov(CMB+noise)')
plt.plot(x, rmse_u_cfn_list, label='U cov(CMB+noise+diffuse fg)')
plt.legend()
plt.xlabel('flux_idx')
plt.ylabel('point source flux density RMSE')
# plt.show()


plt.savefig('/afs/ihep.ac.cn/users/w/wangyiming25/tmp/20230912/cov_u_rmse.png', dpi=300)






