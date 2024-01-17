import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

cov = np.load('./1.npy')
print(f'{cov[:10,:10]=}')
