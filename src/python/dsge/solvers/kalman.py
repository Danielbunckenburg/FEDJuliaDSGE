import numpy as np
from scipy.linalg import solve

def kalman_filter(data, TTT, RRR, QQ, ZZ, DD, EE, s0, P0, outputs=['loglh']):
    """
    Python implementation of Kalman Filter for DSGE.
    Returns loglh by default, or a dict if multiple outputs requested.
    """
    n_obs, T = data.shape
    n_states = TTT.shape[0]
    
    s_filt_history = []
    P_filt_history = []
    s_pred_history = []
    P_pred_history = []
    
    s_filt = s0.copy()
    P_filt = P0.copy()
    
    loglh = 0.0
    
    # Pre-calculate common terms
    RQR = RRR @ QQ @ RRR.T
    
    for t in range(T):
        y_t = data[:, t]
        
        # 1. Predict
        s_pred = TTT @ s_filt
        P_pred = TTT @ P_filt @ TTT.T + RQR
        
        # Store predictions
        s_pred_history.append(s_pred.copy())
        P_pred_history.append(P_pred.copy())
        
        # Handle missing data (NaNs)
        non_missing = ~np.isnan(y_t)
        if not np.any(non_missing):
            s_filt = s_pred
            P_filt = P_pred
        else:
            y_t_sub = y_t[non_missing]
            ZZ_t = ZZ[non_missing, :]
            DD_t = DD[non_missing]
            EE_t = EE[non_missing][:, non_missing]
            
            # 2. Innovation
            v_t = y_t_sub - ZZ_t @ s_pred - DD_t
            F_t = ZZ_t @ P_pred @ ZZ_t.T + EE_t
            
            # 3. Update
            try:
                F_inv_v = solve(F_t, v_t, assume_a='pos')
                F_inv_ZZ_P = solve(F_t, ZZ_t @ P_pred, assume_a='pos')
                
                s_filt = s_pred + (P_pred @ ZZ_t.T) @ F_inv_v
                P_filt = P_pred - (P_pred @ ZZ_t.T) @ F_inv_ZZ_P
                
                # 4. Likelihood
                sign, logdet = np.linalg.slogdet(F_t)
                loglh += -0.5 * (len(v_t) * np.log(2 * np.pi) + logdet + v_t @ F_inv_v)
            except Exception as e:
                print(f"Kalman Filter Error at t={t}: {e}")
                return -1e10 if 'loglh' in outputs else None

        s_filt_history.append(s_filt.copy())
        P_filt_history.append(P_filt.copy())

    results = {}
    if 'loglh' in outputs: results['loglh'] = loglh
    if 'states' in outputs: results['states'] = np.array(s_filt_history).T
    if 'variances' in outputs: results['variances'] = np.array(P_filt_history)
    if 'pred_states' in outputs: results['pred_states'] = np.array(s_pred_history).T
    if 'pred_variances' in outputs: results['pred_variances'] = np.array(P_pred_history)
    
    return results if len(outputs) > 1 else results[outputs[0]]

def kalman_smoother(TTT, RRR, QQ, s_filt_history, P_filt_history, s_pred_history, P_pred_history):
    """
    Rauch-Tung-Striebel (RTS) smoother and Disturbance Smoother for shocks.
    """
    T = s_filt_history.shape[1]
    n_states = s_filt_history.shape[0]
    n_shocks = RRR.shape[1]
    
    s_smooth = np.zeros_like(s_filt_history)
    P_smooth = np.zeros_like(P_filt_history)
    shocks_smooth = np.zeros((n_shocks, T))
    
    s_smooth[:, -1] = s_filt_history[:, -1]
    P_smooth[-1, :, :] = P_filt_history[-1, :, :]
    
    # 1. State Smoothing
    for t in range(T - 2, -1, -1):
        try:
            C_t = solve(P_pred_history[t+1], TTT @ P_filt_history[t], assume_a='pos').T
        except Exception:
            C_t = P_filt_history[t] @ TTT.T @ np.linalg.pinv(P_pred_history[t+1])
            
        s_smooth[:, t] = s_filt_history[:, t] + C_t @ (s_smooth[:, t+1] - s_pred_history[:, t+1])
        P_smooth[t, :, :] = P_filt_history[t, :, :] + C_t @ (P_smooth[t+1, :, :] - P_pred_history[t+1, :, :]) @ C_t.T
    
    # 2. Shock Smoothing (Simplified)
    # epsilon_t = Q * R' * P_pred_t^-1 * (s_smooth_t - s_pred_t)
    for t in range(T):
        diff = s_smooth[:, t] - s_pred_history[:, t]
        try:
            # Q * R' @ inv(P_pred) @ diff
            eps_t = QQ @ RRR.T @ solve(P_pred_history[t], diff, assume_a='pos')
        except Exception:
            eps_t = QQ @ RRR.T @ np.linalg.pinv(P_pred_history[t]) @ diff
        shocks_smooth[:, t] = eps_t
        
    return s_smooth, P_smooth, shocks_smooth


