import ctypes
import numpy as np
import os

class DSGEKernels:
    def __init__(self, lib_path="bin/dsge_kernels.dll"):
        if not os.path.exists(lib_path):
            print(f"Warning: Shared library not found at {lib_path}. Performance kernels will not be available.")
            self.lib = None
            return
        self.lib = ctypes.CDLL(lib_path)
        
        # Define gensys_c signature
        self.lib.gensys_c.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double,
            np.ctypeslib.ndpointer(dtype=np.complex128, ndim=2, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.complex128, ndim=2, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.complex128, ndim=1, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.complex128, ndim=2, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.complex128, ndim=2, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.complex128, ndim=2, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.complex128, ndim=1, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.complex128, ndim=2, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS')
        ]

        # Define kalman_filter_c signature
        self.lib.kalman_filter_c.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS')
        ]

    def solve(self, g0, g1, c, psi, pi, div=0.0):
        if not self.lib: return None
        n = g0.shape[0]
        n_z = psi.shape[1]
        n_eta = pi.shape[1]
        
        g1_out = np.zeros((n, n), dtype=np.complex128)
        c_out = np.zeros(n, dtype=np.complex128)
        impact_out = np.zeros((n, n_z), dtype=np.complex128)
        eu = np.zeros(2, dtype=np.int32)
        
        self.lib.gensys_c(n, n_z, n_eta, div, g0, g1, c, psi, pi, g1_out, c_out, impact_out, eu)
        return g1_out, c_out, impact_out, eu

if __name__ == "__main__":
    print("DSGE Kernels Wrapper loaded.")
