import numpy as np
import pandas as pd

def forecast(model, TTT, RRR, CCC, s_final, horizon=12, shocks=None):
    """
    Produces a forecast starting from the final state s_final.
    
    Args:
        model: Model instance
        TTT, RRR, CCC: Transition matrices
        s_final: Final state vector [n_states_augmented]
        horizon: Quarters to forecast
        shocks: Matrix of future shocks [horizon, n_shocks_exogenous]. 
                If None, defaults to zero shocks.
                
    Returns:
        forecast_states: [horizon, n_states]
        forecast_obs: [horizon, n_obs]
    """
    n_states = TTT.shape[0]
    n_shocks = RRR.shape[1]
    
    if shocks is None:
        shocks = np.zeros((horizon, n_shocks))
        
    states = np.zeros((horizon, n_states))
    
    s_curr = s_final
    for t in range(horizon):
        # s_{t+1} = T * s_t + R * eps_{t+1} + C
        s_next = TTT @ s_curr + RRR @ shocks[t, :] + CCC
        states[t, :] = s_next
        s_curr = s_next
        
    # Map to observables
    ZZ, DD, QQ, EE = model.measurement(TTT, RRR, CCC)
    obs = (ZZ @ states.T).T + DD
    
    return states, obs

def shock_decomposition(model, TTT, RRR, CCC, data):
    """
    Decomposes the observables into the contributions of each shock over time.
    """
    T = data.shape[1]
    n_states = TTT.shape[0]
    n_shocks = RRR.shape[1]
    
    # 1. Get smoothed shocks
    s_smooth, _, eps_smooth = model.smooth(data)
    
    # 2. Decompose
    # Each shock's contribution to states: s_{j,t} = T * s_{j, t-1} + R_j * eps_{j,t}
    # Initial state contribution (if s0 != 0) is excluded or handled as 'initial'
    
    decomp = np.zeros((n_shocks, T, n_states))
    
    for j in range(n_shocks):
        s_j = np.zeros(n_states)
        R_j = RRR[:, j] # shape (n_states,)
        for t in range(T):
            # s_{j,t} = T @ s_{j,t-1} + R_j * eps_{j,t}
            s_j = TTT @ s_j + R_j * eps_smooth[j, t]
            decomp[j, t, :] = s_j
            
    # Map to observables
    ZZ, DD, QQ, EE = model.measurement(TTT, RRR, CCC)
    obs_decomp = np.zeros((n_shocks, T, ZZ.shape[0]))
    for j in range(n_shocks):
        obs_decomp[j, :, :] = (ZZ @ decomp[j, :, :].T).T
        
    return obs_decomp

