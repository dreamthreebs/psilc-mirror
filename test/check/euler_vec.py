import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

eu_mat = hp.rotator.euler_matrix_new(90,90,0, ZYX=True, deg=True)
vec_eu = hp.rotator.rotateVector(eu_mat, vec=[0,0,1])
print(f'{vec_eu=}')

