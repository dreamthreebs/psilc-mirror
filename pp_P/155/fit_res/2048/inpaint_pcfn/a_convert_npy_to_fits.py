import numpy as np
import healpy as hp
from concurrent.futures import ProcessPoolExecutor

def process_realization(rlz_idx, freq, threshold):
    print(f'Processing realization index: {rlz_idx}')
    m = np.load(f'../../../../../fitdata/synthesis_data/2048/PSCMBFGNOISE/{freq}/{rlz_idx}.npy')
    hp.write_map(f'./{threshold}sigma/input/Q/{rlz_idx}.fits', m[1], overwrite=True)
    hp.write_map(f'./{threshold}sigma/input/U/{rlz_idx}.fits', m[2], overwrite=True)

freq = 155
threshold = 3
with ProcessPoolExecutor() as executor:
    executor.map(process_realization, range(100), [freq]*100, [threshold]*100)

