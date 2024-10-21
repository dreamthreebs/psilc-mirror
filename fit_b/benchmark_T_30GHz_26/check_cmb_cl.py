import numpy as np


cls = np.load('../../src/cmbsim/cmbdata/cmbcl_8k.npy').T
print(f'{cls[0,0:100]=}')

