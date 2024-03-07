import numpy as np
import sys
from memory_profiler import profile

class y:
    def __init__(self, m):
        self.m = m

    def run(self):
        y = self.m[0:100]
        return y + 100

@profile
def my_loop():
    for flux_idx in range(5):

        print(f'{flux_idx=}')
        m = np.load(f'../../fitdata/synthesis_data/2048/CMBFGNOISE/40/{flux_idx}.npy')[0]
        print(f'{sys.getrefcount(m)-1=}')
        obj = y(m)
        print(f'{sys.getrefcount(m)-1=}')
        x = obj.run()
        print(f'{sys.getrefcount(m)-1=}')
        z = x+200
        print(f'{sys.getrefcount(m)-1=}')


my_loop()
