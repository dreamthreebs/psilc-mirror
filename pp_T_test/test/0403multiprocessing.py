# from multiprocessing import Pool
import time
import numpy as np



def funcMIne(_):
    for i in range(100):

        time.sleep(5)
        # np.sum(np.random.rand(100000000))
        print(f'{i=}')
    return 1

funcMIne(0)

#with Pool(processes=5) as pool:
#    # Map simulateParticle function to the arguments for each particle
#    results = pool.map(funcMIne, range(100))
#    #results = list(tqdm(pool.imap_unordered(self._wrapperRW, range(self.num_particles)), total=self.num_particles)) #list is used to turn the iterator into a list, since we use pool.imap_unordered instead of pool.map
