import numpy as np
from .solvers.gensys import gensys
from .solvers.kalman import kalman_filter

from scipy.linalg import solve_discrete_lyapunov

def likelihood(m, data):
    """
    Computes the log-likelihood of the model given the data.
    """
    # 1. Solve the model
    TTT, RRR, CCC = m.solve()
    
    # 2. Extract measurement matrices
    ZZ, DD, QQ, EE = m.measurement(TTT, RRR, CCC)
    
    # 3. Initial values (stationary distributions)
    s0, P0 = m.init_stationary_states(TTT, RRR, QQ)
    
    # 4. Kalman Filter
    # data expects [n_obs, T]
    loglh = kalman_filter(data, TTT, RRR, QQ, ZZ, DD, EE, s0, P0)
    
    return loglh
