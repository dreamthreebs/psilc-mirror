import numpy as np
var1 = np.load('./fit_qu_no_const/idx_0/fit_P_0.npy')
var2 = np.load('./fit_qu_1024/idx_0/fit_P_0.npy')
var3 = np.load('./fit_qu_no_const/idx_0/P_0.npy')
var4 = np.load('./fit_qu_1024/idx_0/P_0.npy')
var5 = np.load('./fit_qu_no_const/idx_0/fit_err_P_0.npy')
var6 = np.load('./fit_qu_1024/idx_0/fit_err_P_0.npy')



print(f'{var1=}, {var2=}, {var3=}, {var4=}, {var5=}, {var6=}')
