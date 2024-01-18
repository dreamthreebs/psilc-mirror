import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../FreqBand')

def main():
    freq = 40
    print(f'{freq=}')

    strongps = hp.read_map(f'/sharefs/alicpt/users/zrzhang/allFreqPSMOutput/observations/AliCPT_uKCMB/{freq}GHz/group3_map_{freq}GHz.fits', field=(0,1,2))
    print(f'{hp.get_nside(strongps)=}')

    diffusefg = hp.read_map(f'/sharefs/alicpt/users/zrzhang/allFreqPSMOutput/observations/AliCPT_uKCMB/{freq}GHz/group1_map_{freq}GHz.fits', field=(0,1,2))

    cmb = np.load(f'../../inpaintingdata/CMBREALIZATION/40GHz/0.npy')
    # hp.mollview(ps[0], norm='hist', title='ps2048')
    # plt.show()

    # hp.mollview(ps512[0], norm='hist', title='ps512')
    # plt.show()

    nstd = np.load(f'../NSTDNORTH/2048/{freq}.npy')
    n_pix_nstd = hp.get_map_size(nstd[0])

    noise = nstd * np.random.normal(0,1,(3,n_pix_nstd))

    # hp.mollview(noise, title=f'noise at {freq=}')
    # plt.show()

    PSCMBFGNOISE = strongps + diffusefg + noise + cmb
    PSCMB = strongps + cmb
    PSCMBNOISE = strongps + cmb + noise
    PSFGNOISE = strongps + diffusefg + noise
    PSNOISE = strongps + noise

    PSCMBFG = strongps + cmb +diffusefg

    FGNOISE = diffusefg + noise
    CMBFG = cmb + diffusefg
    CMBFGNOISE = cmb + diffusefg + noise


    # hp.mollview(PSCMBFGNOISE[0], title=f'{freq=}', norm='hist')

    np.save(f'./PSCMBFGNOISE/{freq}.npy', PSCMBFGNOISE)
    np.save(f'./PSCMB/{freq}.npy', PSCMB)
    np.save(f'./PSCMBNOISE/{freq}.npy', PSCMBNOISE)
    np.save(f'./PSFGNOISE/{freq}.npy', PSFGNOISE)
    np.save(f'./PSNOISE/{freq}.npy', PSNOISE)
    np.save(f'./FGNOISE/{freq}.npy', FGNOISE)
    np.save(f'./CMBFG/{freq}.npy', CMBFG)
    np.save(f'./CMBFGNOISE/{freq}.npy', CMBFGNOISE)
    np.save(f'./PSCMBFG/{freq}.npy', PSCMBFG)

main()


