import numpy as np
import healpy as hp
from concurrent.futures import ProcessPoolExecutor
import os

def process_realization(rlz_idx, freq, threshold):
    print(f'{rlz_idx=}')
    file_path = f'../../../../../fitdata/synthesis_data/2048/PSCMBFGNOISE/{freq}/{rlz_idx}.npy'
    m = np.load(file_path)
    hp.write_map(f'./{threshold}sigma/input/Q/{rlz_idx}.fits', m[1], overwrite=True)
    hp.write_map(f'./{threshold}sigma/input/U/{rlz_idx}.fits', m[2], overwrite=True)

freq = 145
threshold = 3
max_workers = 4  # Limit to avoid excessive concurrent disk I/O
with ProcessPoolExecutor(max_workers=max_workers) as executor:
    executor.map(process_realization, range(100), [freq]*100, [threshold]*100)

