import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('../FGSim/FreqBand5')

print(f'{len(df)=}')
nside_out = 512

# for i in range(len(df)):
#     freq = df.at[i, 'freq']
#     beam = df.at[i, 'beam']
#     print(f'{freq=},{beam=}')
#     m = hp.read_map(f'/sharefs/alicpt/users/zrzhang/allFreqPSMOutput/observations/AliCPT_uKCMB/{freq}GHz/group1_map_{freq}GHz.fits', field=(0,1,2))
#     m_out = hp.ud_grade(m, nside_out=nside_out)
#     np.save(f'./diffusefg/{freq}.npy', m_out.astype(np.float32))

# for i in range(len(df)):
#     freq = df.at[i, 'freq']
#     beam = df.at[i, 'beam']
#     print(f'{freq=},{beam=}')
#     m = hp.read_map(f'/sharefs/alicpt/users/zrzhang/allFreqPSMOutput/observations/AliCPT_uKCMB/{freq}GHz/group2_map_{freq}GHz.fits', field=(0,1,2))
#     m_out = hp.ud_grade(m, nside_out=nside_out)
#     np.save(f'./faintps/{freq}.npy', m_out.astype(np.float32))

for i in range(len(df)):
    freq = df.at[i, 'freq']
    beam = df.at[i, 'beam']
    print(f'{freq=},{beam=}')
    m = hp.read_map(f'/sharefs/alicpt/users/zrzhang/allFreqPSMOutput/observations/AliCPT_uKCMB/{freq}GHz/group3_map_{freq}GHz.fits', field=(0,1,2))
    hp.mollview(m[0])
    plt.show()
    # m_out = hp.ud_grade(m, nside_out=nside_out)
    # np.save(f'./strongps/{freq}.npy', m_out.astype(np.float32))

# for i in range(len(df)):
#     freq = df.at[i, 'freq']
#     beam = df.at[i, 'beam']
#     print(f'{freq=},{beam=}')
#     m = hp.read_map(f'/sharefs/alicpt/users/zrzhang/allFreqPSMOutput/observations/AliCPT_uKCMB/{freq}GHz/group4_map_{freq}GHz.fits', field=(0,1,2))
#     m_out = hp.ud_grade(m, nside_out=nside_out)
#     np.save(f'./allcomponent/{freq}.npy', m_out.astype(np.float32))

# for i in range(len(df)):
#     freq = df.at[i, 'freq']
#     beam = df.at[i, 'beam']
#     print(f'{freq=},{beam=}')
#     m = hp.read_map(f'/sharefs/alicpt/users/zrzhang/allFreqPSMOutput/observations/AliCPT_uKCMB/{freq}GHz/group1_map_{freq}GHz.fits', field=(0,1,2)) + hp.read_map(f'/sharefs/alicpt/users/zrzhang/allFreqPSMOutput/observations/AliCPT_uKCMB/{freq}GHz/group3_map_{freq}GHz.fits', field=(0,1,2))
#     m = hp.read_map(f'/sharefs/alicpt/users/zrzhang/allFreqPSMOutput/observations/AliCPT_uKCMB/{freq}GHz/group1_map_{freq}GHz.fits', field=(0,1,2)) + hp.read_map(f'/sharefs/alicpt/users/zrzhang/allFreqPSMOutput/observations/AliCPT_uKCMB/{freq}GHz/group3_map_{freq}GHz.fits', field=(0,1,2))
#     # m_out = hp.ud_grade(m, nside_out=nside_out)
#     # np.save(f'./STRONGDIFFUSE/{freq}.npy', m_out.astype(np.float32))








