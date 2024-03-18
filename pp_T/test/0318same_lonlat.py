import numpy as np
import healpy as hp

dir1 = (30,60)
dir2 = (30,60)

ang = hp.rotator.angdist(dir1=dir1, dir2=dir2, lonlat=True)
print(f'{ang=}')

