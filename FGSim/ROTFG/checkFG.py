import healpy as hp
import numpy as np
import glob
mask = np.load("../../mask/512/SNMask/northMask_for_ilc.npy")

fl = glob.glob("*.npy")

def TQU2B(m):
    nside = hp.get_nside(m)
    almt, alme, almb = hp.map2alm(m)
    m = hp.alm2map(almb, nside)
    return m

for name in fl:
    m1 = TQU2B(np.load("../FG/"+name))
    m2 = TQU2B(np.load(name))
    print(name)
    print("FG", np.std(m1[mask]))
    print("ROTFG", np.std(m2[mask]))
    print("\n")


