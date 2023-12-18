import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

vec = (0,0,1)
r = hp.rotator.Rotator(rot=[45, 45, 0])
r_vec = r(0,0,1)
print(r_vec)

r_vec = r(0,1,0)
print(r_vec)

r_vec = r(1,0,0)
print(r_vec)
