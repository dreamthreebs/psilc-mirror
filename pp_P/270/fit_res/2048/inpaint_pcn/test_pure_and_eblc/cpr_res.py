import numpy as np
import matplotlib.pyplot as plt

lmax = 1500
l = np.arange(lmax+1)

ell_arr = np.load('./dl_data/ell_arr.npy')
# dl_theory = np.load('./dl_data/dl_theory.npy')
cl = np.load('../../../../../../src/cmbsim/cmbdata/cmbcl.npy').T

dl_theory = l*(l+1)*cl[:,0:lmax+1] / (2*np.pi)

# plt.plot(l, dl_theory[0])
# plt.plot(l, dl_theory[1])
# plt.plot(l, dl_theory[2])
# plt.semilogy()
# plt.show()


cln_b_list = []
crt_e_list = []
no_pure_e_list = []
no_pure_b_list = []
pure_b_list = []
no_lkg_list = []
delta_1_list = []
delta_2_list = []

for rlz_idx in range(100):
    cln_b = np.load(f'./dl_data/cln_b/{rlz_idx}.npy')
    crt_e = np.load(f'./dl_data/crt_e/{rlz_idx}.npy')
    no_pure_e = np.load(f'./dl_data/no_pure_e/{rlz_idx}.npy')
    no_pure_b = np.load(f'./dl_data/no_pure_b/{rlz_idx}.npy')
    pure_b = np.load(f'./dl_data/pure_b/{rlz_idx}.npy')
    no_lkg = np.load(f'./dl_data/no_lkg/{rlz_idx}.npy')
    delta_1 = pure_b - no_lkg
    delta_2 = cln_b - no_lkg

    cln_b_list.append(cln_b)
    crt_e_list.append(crt_e)
    no_pure_e_list.append(no_pure_e)
    no_pure_b_list.append(no_pure_b)
    pure_b_list.append(pure_b)
    no_lkg_list.append(no_lkg)
    delta_1_list.append(delta_1)
    delta_2_list.append(delta_2)

cln_b_arr = np.array(cln_b_list)
crt_e_arr = np.array(crt_e_list)
no_pure_e_arr = np.array(no_pure_e_list)
no_pure_b_arr = np.array(no_pure_b_list)
pure_b_arr = np.array(pure_b_list)
no_lkg_arr = np.array(no_lkg_list)
delta_1_arr = np.array(delta_1_list)
delta_2_arr = np.array(delta_2_list)

cln_b_mean = np.mean(cln_b_arr, axis=0)
crt_e_mean = np.mean(crt_e_arr, axis=0)
no_pure_e_mean = np.mean(no_pure_e_arr, axis=0)
no_pure_b_mean = np.mean(no_pure_b_arr, axis=0)
pure_b_mean = np.mean(pure_b_arr, axis=0)
no_lkg_mean = np.mean(no_lkg_arr, axis=0)

cln_b_std = np.std(cln_b_arr, axis=0)
crt_e_std = np.std(crt_e_arr, axis=0)
no_pure_e_std = np.std(no_pure_e_arr, axis=0)
no_pure_b_std = np.std(no_pure_b_arr, axis=0)
pure_b_std = np.std(pure_b_arr, axis=0)
no_lkg_std = np.std(no_lkg_arr, axis=0)
delta_1_std = np.std(delta_1_arr, axis=0)
delta_2_std = np.std(delta_2_arr, axis=0)


plt.figure(1)
# plt.plot(l, dl_theory[0], label='TT theory')

# plt.plot(l, dl_theory[1], label='EE theory')
# plt.plot(l, dl_theory[2], label='BB theory')

plt.plot(ell_arr, cln_b_mean, label='cln_b')
plt.plot(ell_arr, crt_e_mean, label='crt_e')
plt.plot(ell_arr, no_pure_e_mean, label='no_pure_e')
plt.plot(ell_arr, no_pure_b_mean, label='no_pure_b')
plt.plot(ell_arr, pure_b_mean, label='pure_b')
plt.plot(ell_arr, no_lkg_mean, label='no lkg b')

plt.loglog()
plt.legend()

plt.xlabel('$\\ell$')
plt.ylabel('$D_\\ell[\\mu K^2]$')

plt.figure(2)

plt.plot(ell_arr, cln_b_std, label='cln_b')
plt.plot(ell_arr, crt_e_std, label='crt_e')
plt.plot(ell_arr, no_pure_e_std, label='no_pure_e')
plt.plot(ell_arr, no_pure_b_std, label='no_pure_b')
plt.plot(ell_arr, pure_b_std, label='pure_b')
plt.plot(ell_arr, no_lkg_std, label='no lkg b')

# plt.plot(l, np.sqrt(2/(2*l+1)*dl_theory[2]), label='pure_b')
plt.loglog()
plt.legend(loc='upper right')
plt.xlabel('$\\ell$')
plt.ylabel('$D_\\ell[\\mu K^2]$')

plt.figure(3)
plt.plot(ell_arr, delta_1_std, label='master + PURE')
plt.plot(ell_arr, delta_2_std, label='master + recycling method')
plt.loglog()
plt.legend()
plt.show()


