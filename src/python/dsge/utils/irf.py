import numpy as np
import pandas as pd

def compute_irf(model, TTT, RRR, CCC, shock_idx, horizon=20, shock_size=1.0):
    """
    Computes IRFs for a specific shock.
    """
    n_states = TTT.shape[0]
    n_shocks = RRR.shape[1]
    
    # Initialize states
    states = np.zeros((horizon, n_states))
    
    # Impact period
    # s_1 = T s_0 + R eps_1 + C
    # Assume steady state start s_0 (already accounted for in RRR/TTT/CCC usually, 
    # but we look at deviations)
    
    eps = np.zeros(n_shocks)
    eps[shock_idx] = shock_size
    
    states[0, :] = RRR @ eps
    
    for t in range(1, horizon):
        states[t, :] = TTT @ states[t-1, :]
        
    return states

def irf_to_df(model, irf_matrix, observables=True, TTT=None, RRR=None, CCC=None):
    """
    Converts state IRFs to Observables if requested.
    """
    if observables:
        ZZ, DD, QQ, EE = model.measurement(TTT, RRR, CCC)
        # ZZ is (n_obs, n_states)
        obs_irf = (ZZ @ irf_matrix.T).T
        df = pd.DataFrame(obs_irf, columns=list(model.observables.keys()))
    else:
        df = pd.DataFrame(irf_matrix, columns=list(model.endogenous_states.keys()))
        
    df.index.name = "Quarter"
    return df
