import numpy as np
import matplotlib.pyplot as plt

cl1 = np.load('./cmbdata/cmbcl.npy')
cl2 = np.load('./cmbdata/cmbcl_8k.npy')

l1 = np.arange(2000)
l2 = np.arange(8001)

dl1 = l1*(l1+1) *cl1[:,2]/(2*np.pi)
dl2 = l2*(l2+1) *cl2[:,2]/(2*np.pi)

plt.loglog(l1, dl1)
plt.loglog(l2, dl2)
plt.show()
