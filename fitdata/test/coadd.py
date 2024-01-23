import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

def add_then_degrade():
    cmb = np.load('../2048/CMB/40/0.npy')
    ps = np.load('../2048/PS/40/ps.npy')
    fg = np.load('../2048/FG/40/fg.npy')
    
    m = cmb + ps
    ud_m = hp.ud_grade(m, nside_out=256)

    return ud_m

def degrade_then_add():
    cmb = np.load('../256/CMB/40/0.npy')
    ps = np.load('../256/PS/40/ps.npy')
    fg = np.load('../256/FG/40/fg.npy')

    ud_m = cmb + ps

    return ud_m

m1 = add_then_degrade()[0]
m2 = degrade_then_add()[0]

hp.mollview(m1)
hp.mollview(m2)
hp.mollview(m1-m2)
plt.show()


