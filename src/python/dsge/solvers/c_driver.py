import os
import ctypes
import numpy as np
import scipy.linalg.cython_blas
import scipy.linalg.cython_lapack

# Load the C library
# Try to find the .dll in the same directory or src/c/dsge_c_lib.dll
_c_lib = None

def load_c_lib():
    global _c_lib
    if _c_lib:
        return _c_lib
        
    # Search paths
    base_dir = os.path.dirname(os.path.abspath(__file__)) # dsge/solvers
    # Expected relative path to src/c/dsge_c_lib.dll if running from dev setup
    # But usually we might have compiled it into src/c/
    
    # Try multiple likely locations
    paths = [
        os.path.join(base_dir, "../../../c/dsge_c_lib.dll"), # src/python/dsge/solvers -> src/c/
        os.path.join(base_dir, "../../../c/dsge_c_lib.so"),
        "./dsge_c_lib.dll",
        "./dsge_c_lib.so"
    ]
    
    lib_path = None
    for p in paths:
        if os.path.exists(p):
            lib_path = p
            break
            
    if not lib_path:
        # Fallback to absolute path construction based on project root if possible
        # Project root usually 3 levels up from here
        root = os.path.abspath(os.path.join(base_dir, "../../../"))
        c_path_win = os.path.join(root, "src", "c", "dsge_c_lib.dll")
        if os.path.exists(c_path_win):
            lib_path = c_path_win
            
    if not lib_path:
        print("Warning: C extension not found. Using pure Python fallbacks.")
        return None
        
    try:
        _c_lib = ctypes.CDLL(lib_path)
        _setup_lapack(_c_lib)
        print(f"Loaded C extension from {lib_path}")
    except OSError as e:
        print(f"Failed to load C extension: {e}")
        return None
        
    return _c_lib

def _get_capsule_address(capsule, name):
    # Extract address from PyCapsule (used by Cython) or __pyx_capi__
    # Scipy < 1.10 might differ, but generally __pyx_capi__ is a dict of names to PyCapsules
    pass
    
def _setup_lapack(lib):
    # Extract function pointers from scipy
    # scipy.linalg.cython_blas.__pyx_capi__ is a dict
    
    def get_addr(module, name):
        capi = getattr(module, "__pyx_capi__", {})
        if name in capi:
            return ctypes.pythonapi.PyCapsule_GetPointer(ctypes.py_object(capi[name]), None)
        return None

    dgemm_addr = get_addr(scipy.linalg.cython_blas, "dgemm")
    dgemv_addr = get_addr(scipy.linalg.cython_blas, "dgemv")
    dcopy_addr = get_addr(scipy.linalg.cython_blas, "dcopy")
    dposv_addr = get_addr(scipy.linalg.cython_lapack, "dposv")
    zgges_addr = get_addr(scipy.linalg.cython_lapack, "zgges")
    
    if not all([dgemm_addr, dgemv_addr, dposv_addr, zgges_addr]):
        print("Warning: Could not find all BLAS/LAPACK symbols in Scipy.")
        return

    # Define setup_lapack arg types
    # void setup_lapack(void*, void*, void*, void*, void*)
    lib.setup_lapack.argtypes = [ctypes.c_void_p] * 5
    lib.setup_lapack.restype = None
    
    lib.setup_lapack(dgemm_addr, dgemv_addr, dposv_addr, dcopy_addr, zgges_addr)

# Helper for argument preparation
def _as_complex_ptr(arr):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) # Complex is 2 doubles, but we pass pointer

def gensys_c(g0, g1, c, psi, pi):
    lib = load_c_lib()
    if not lib:
        return None # Signal fallback
        
    n = g0.shape[0]
    
    # Ensure inputs are complex128 contiguous (Fortran-style for LAPACK usually preferred but we used RowMajor malloc in C?)
    # Wait, our C code manually converts to A and B using malloc loops.
    # The inputs g0, g1... are passed as pointers.
    # Our C code expects complex*
    
    g0_c = np.ascontiguousarray(g0, dtype=np.complex128)
    g1_c = np.ascontiguousarray(g1, dtype=np.complex128)
    c_c = np.ascontiguousarray(c, dtype=np.complex128)
    psi_c = np.ascontiguousarray(psi, dtype=np.complex128)
    pi_c = np.ascontiguousarray(pi, dtype=np.complex128)
    
    G1 = np.zeros_like(g0_c)
    C_out = np.zeros_like(c_c)
    impact = np.zeros((n, psi.shape[1]), dtype=np.complex128)
    eu = np.zeros(2, dtype=np.int32)
    
    # gensys_c(int n, complex *g0, complex *g1, ..., int *eu)
    lib.gensys_c.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, 
        ctypes.POINTER(ctypes.c_int)
    ]
    
    lib.gensys_c(
        n,
        g0_c.ctypes.data, g1_c.ctypes.data, c_c.ctypes.data, psi_c.ctypes.data, pi_c.ctypes.data,
        G1.ctypes.data, C_out.ctypes.data, impact.ctypes.data,
        eu.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    )
    
    if eu[0] == 1 and eu[1] == 1:
        return G1, C_out, impact, eu
    else:
        return None # Failed or not fully implemented

def kalman_filter_c(data, TTT, RRR, QQ, ZZ, DD, EE, s0, P0):
    lib = load_c_lib()
    if not lib:
        return None

    # Dimensions
    n_obs, T = data.shape
    n_states = TTT.shape[0]
    n_shocks = RRR.shape[1]
    
    # Prepare types (double)
    data_c = np.ascontiguousarray(data, dtype=np.float64) # F-order?
    # Our C code uses manual indexing: data[t*n_obs + i]. This implies Row Major if we iterate T then n_obs
    # wait. "memcpy(v, data + t * n_obs, n_obs ...)" -> This assumes data is flat: [obs_1_t1, obs_2_t1, ..., obs_1_t2...]
    # This is Column-Major (Fortran) if shape is (n_obs, T).
    # Numpy default is Row-Major (C-style): [obs_1_t1, obs_1_t2 ...]. No wait.
    # Numpy (n_obs, T) row-major:
    #   Row 0: obs_1_t1, obs_1_t2, ...
    #   Row 1: obs_2_t1, obs_2_t2, ...
    # If we want time-contiguous blocks, we want F-order (Column Major) for (n_obs, T).
    # So `data + t*n_obs` lands on `t`-th column? No.
    # In F-order:
    #   Col 0 (t=0): obs_1_t0, obs_2_t0...
    #   Col 1 (t=1): obs_1_t1, obs_2_t1...
    # So memory is [obs_1_t0, obs_2_t0, ..., obs_1_t1...]
    # This matches `data + t*n_obs` offset if we step by n_obs.
    # So YES, use F-order.
    
    data_c = np.asfortranarray(data, dtype=np.float64)
    TTT_c = np.asfortranarray(TTT, dtype=np.float64)
    RRR_c = np.asfortranarray(RRR, dtype=np.float64)
    QQ_c = np.asfortranarray(QQ, dtype=np.float64)
    ZZ_c = np.asfortranarray(ZZ, dtype=np.float64)
    DD_c = np.asfortranarray(DD, dtype=np.float64)
    EE_c = np.asfortranarray(EE, dtype=np.float64)
    s0_c = np.asfortranarray(s0, dtype=np.float64)
    P0_c = np.asfortranarray(P0, dtype=np.float64)
    
    loglh = ctypes.c_double(0.0)
    
    lib.kalman_filter_c.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double)
    ]
    
    lib.kalman_filter_c(
        n_obs, n_states, n_shocks, T,
        data_c.ctypes.data, TTT_c.ctypes.data, RRR_c.ctypes.data, QQ_c.ctypes.data,
        ZZ_c.ctypes.data, DD_c.ctypes.data, EE_c.ctypes.data, s0_c.ctypes.data, P0_c.ctypes.data,
        ctypes.byref(loglh)
    )
    
    return loglh.value
