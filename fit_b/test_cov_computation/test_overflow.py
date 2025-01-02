from ctypes import cdll, c_void_p
import numpy as np

# Load the library
lib = cdll.LoadLibrary('./test.so')

npix = 20000
covmat = np.zeros((3*npix, 3*npix), dtype=np.float64)
# Declare the function signature
# lib.checkPointerOverflow.restype = None  # Returns void

# Call the function
lib.checkPointerOverflow(c_void_p(covmat.ctypes.data))

