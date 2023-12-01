import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

m = np.load('./2048.npy')
nside_out = 512
m_out = hp.ud_grade(m, nside_out=nside_out)
np.save('./d_2048_512.npy', m_out)
