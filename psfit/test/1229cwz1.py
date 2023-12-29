import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pymaster as nmt


nside=1024
mask = hp.read_map('/sharefs/alicpt/users/chenwz/Testarea/PARTIAL_SKY/AliCPT_20uKcut150_C_1024.fits', field=0)
sm_mask = hp.read_map('/sharefs/alicpt/users/chenwz/Testarea/PARTIAL_SKY/mask_1024_Sm.fits',field=0)
m = hp.read_map('/sharefs/alicpt/users/chenwz/Testarea/PARTIAL_SKY/lensed_1024_map.fits',field=(0,1,2))
print(f'{mask.shape=}')
print(f'{m.shape=}')

lmax=1000
l = np.arange(lmax+1)
cl = hp.anafast(m, lmax=lmax)


