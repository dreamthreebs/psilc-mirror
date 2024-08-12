import numpy as np
import glob

noise_path = glob.glob('./noise_sim/*.npy')
cmb_path = glob.glob('./cmb_sim/*.npy')
n_list = []
c_list = []
for p_n, p_c in zip(noise_path, cmb_path):
    n = np.load(p_n)
    cmb = np.load(p_c)

    n_list.append(n)
    c_list.append(cmb)

n_arr = np.asarray(n_list)
c_arr = np.asarray(c_list)
cov = np.cov(n_arr, rowvar=False)
cmb_cov = np.load('./cmb_b_cov/0.npy')
c_cov = np.cov(c_arr, rowvar=False)

# print(f'{cov - cov.T}')
print(f'{c_cov=}')
np.set_printoptions(threshold=np.inf)
print(f'{cmb_cov=}')
hp.
# print(f'{cov=}')
# print(f'{cmb_cov=}')

print(f'{cov.shape=}')
np.save('./data/noise_cov.npy', cov)
np.save('./data/c_cov.npy', c_cov)



