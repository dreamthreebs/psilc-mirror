import healpy as hp
import numpy as np
import glob
import os

fl = glob.glob("../FG/*.npy")

r = hp.Rotator(coord='GC')

for name in fl:
    m = np.load(name)
    m = r.rotate_map_pixel(m)
    outname = os.path.basename(name)
    np.save(outname, m)
    print(name)
