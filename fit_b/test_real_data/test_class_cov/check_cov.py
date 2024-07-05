import numpy as np

# cmb_class_cov = np.load('./cmb.npy')
# cmb_func_cov = np.load('../cmb_b_cov/1.npy')
# np.set_printoptions(threshold=np.inf)
# print(f'{cmb_class_cov-cmb_func_cov=}')

noise_class_cov = np.load('./noise.npy')
noise_func_cov = np.load('../noise_b_cov/1.npy')
np.set_printoptions(threshold=np.inf)
print(f'{noise_class_cov-noise_func_cov=}')
