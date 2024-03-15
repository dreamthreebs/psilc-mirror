import numpy as np
import healpy as hp
import pickle
import matplotlib.pyplot as plt


with open('./cs/cs.pkl', 'rb') as f:
    cs = pickle.load(f)

# with open('./cs350/cs.pkl', 'rb') as f1:
#     cs1 = pickle.load(f1)

cos_theta = np.linspace(0.99,1, 1000)
cov = cs(cos_theta)
# cov1 = cs1(cos_theta)

plt.plot(cos_theta, cov, label='lmax=1999')
# plt.plot(cos_theta, cov1, label='lmax=350')
plt.legend()
plt.show()

