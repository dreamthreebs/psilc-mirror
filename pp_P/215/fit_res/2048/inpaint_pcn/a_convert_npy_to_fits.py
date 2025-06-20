import numpy as np
import healpy as hp

freq = 215
threshold = 3
for rlz_idx in range(100):
    print(f'{rlz_idx=}')
    m = np.load(f'../../../../../fitdata/synthesis_data/2048/PSCMBNOISE/{freq}/{rlz_idx}.npy')
    hp.write_map(f'./{threshold}sigma/input/Q/{rlz_idx}.fits', m[1], overwrite=True)
    hp.write_map(f'./{threshold}sigma/input/U/{rlz_idx}.fits', m[2], overwrite=True)
