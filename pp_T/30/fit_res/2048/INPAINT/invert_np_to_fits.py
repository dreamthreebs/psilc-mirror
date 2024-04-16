import numpy as np
import healpy as hp
from multiprocessing import Pool
from pathlib import Path

# for rlz_idx in range(100):
#     print(f'{rlz_idx=}')
#     m = np.load(f'../../../../../../fitdata/synthesis_data/2048/PSCMBFGNOISE/155/{rlz_idx}.npy')[0].copy()
#     hp.write_map(f'./pcfn/{rlz_idx}.fits', m, overwrite=True)

# Function to be executed in parallel
freq = 30

path_fits = Path(f'./input/pcfn/2sigma')

path_fits.mkdir(exist_ok=True, parents=True)
def process_file(rlz_idx):
    print(f'{rlz_idx=}')
    m = np.load(f'../../../../../fitdata/synthesis_data/2048/PSCMBFGNOISE/{freq}/{rlz_idx}.npy')[0].copy()
    hp.write_map(path_fits / Path(f"{rlz_idx}.fits"), m, overwrite=True)

# Number of processes to use, ideally not more than the number of cores in your machine
num_processes = 20

# Creating a pool of processes and mapping the function to the inputs
if __name__ == '__main__':  # This check is necessary for multiprocessing to work on Windows
    with Pool(num_processes) as pool:
        pool.map(process_file, range(0,100))



