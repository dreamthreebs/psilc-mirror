import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from scipy.special import legendre
import pandas as pd
from numpy.polynomial.legendre import Legendre
import time
import pickle
# from numba import jit

def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record the start time
        result = func(*args, **kwargs)  # Call the function
        end_time = time.time()  # Record the end time
        print(f"Function {func.__name__} took {end_time - start_time} seconds to execute.")
        return result
    return wrapper

lmax = 1500
def calc_C_theta_np(x, lmax, cl):
    sum_val = 0.0
    for l in range(lmax + 1):
        # Create coefficients for the Legendre polynomial of degree l
        # All coefficients are 0 except the one at index l which is 1
        coeffs = [0] * l + [1]
        # Create the Legendre polynomial using the coefficients
        Pl = Legendre(coeffs)
        # Evaluate the polynomial at x and add to the sum
        sum_val += (2 * l + 1) * cl[l] * Pl(x)
    return sum_val

@timeit
def calc_C_theta(x, lmax, cl):
    legendre_polys = [Legendre([0]*l + [1])(x) for l in range(lmax + 1)]
    coefficients = (2 * np.arange(lmax + 1) + 1) * cl
    sum_val = np.dot(coefficients, legendre_polys)
    return sum_val

def calc_C_theta_scipy(x, lmax, cl):
    sum_val = 0.0
    for l in range(lmax + 1):
        Pl = legendre(l)  # Legendre polynomial of degree l
        sum_val += (2 * l + 1) * cl[l] * Pl(x)
    return sum_val

def calc_C_theta_nb(x, lmax, cl):
    sum_val = 0.0
    for l in range(lmax + 1):
        # Create coefficients for the Legendre polynomial of degree l
        coeffs = [0] * l + [1]
        # Create the Legendre polynomial using the coefficients
        Pl = Legendre(coeffs)
        # Evaluate the polynomial at x and add to the sum
        sum_val += (2 * l + 1) * cl[l] * Pl(x)
    return sum_val

legendre_cache = {}

@timeit
def calc_C_theta_cache(x, lmax, cl):
    global legendre_cache
    sum_val = 0.0

    for l in range(lmax + 1):
        # Check if the polynomial is already in the cache
        if l not in legendre_cache:
            coeffs = [0] * l + [1]
            Pl = Legendre(coeffs)
            legendre_cache[l] = Pl
        else:
            Pl = legendre_cache[l]

        # Evaluate the polynomial at x and add to the sum
        sum_val += (2 * l + 1) * cl[l] * Pl(x)

    return sum_val

def evaluate_interp_func(l, x, interp_funcs):
    for interp_func, x_range in interp_funcs[l]:
        if x_range[0] <= x <= x_range[1]:
            return interp_func(x)
    raise ValueError(f"x = {x} is out of the interpolation range for l = {l}")

@timeit
def calc_C_theta_itp(x, lmax, cl, itp_funcs):
    sum_val = 0.0
    for l in range(lmax + 1):
        sum_val += (2 * l + 1) * cl[l] * evaluate_interp_func(l, x, interp_funcs=itp_funcs)

    return sum_val

@timeit
def open_pkl_file():
    with open('../interpolate_cov/lgd_itp_funcs1500.pkl', 'rb') as f:
        loaded_itp_funcs = pickle.load(f)

# open_pkl_file()
cl = np.load('../../src/cmbsim/cmbdata/cmbcl.npy')[:,0]

y = calc_C_theta(0.99997, lmax=lmax, cl=cl[0:lmax+1])
# y1 = calc_C_theta_np(1.0, lmax=lmax, cl=cl[0:lmax+1])
# y2 = calc_C_theta_scipy(1.0, lmax=lmax, cl=cl[0:lmax+1])
# y3 = calc_C_theta_nb(1.0, lmax=lmax, cl=cl[0:lmax+1])
y4 = calc_C_theta_cache(0.99997, lmax=lmax, cl=cl[0:lmax+1])

with open('../interpolate_cov/lgd_itp_funcs1500.pkl', 'rb') as f:
    loaded_itp_funcs = pickle.load(f)
    y5 = calc_C_theta_itp(0.99997, lmax=lmax, cl=cl[0:lmax+1], itp_funcs=loaded_itp_funcs)

print(f'optimized numpy legendre {y=} ')
# print(f'numpy legendre {y1=}')
# print(f'scipy legendre {y2=}')
# print(f'numba legendre {y3=}')
print(f'cache legendre {y4=}')
print(f'interpolated legendre {y5=}')



