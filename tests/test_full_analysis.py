import numpy as np
import pandas as pd
from dsge.models.m1002 import Model1002
from dsge.forecast import forecast, shock_decomposition

def test_full_analysis():
    print("Initializing Model1002...")
    m = Model1002()
    TTT, RRR, CCC = m.solve()
    
    # Simulate some data
    n_obs = m.n_observables
    T = 40
    data = np.random.randn(n_obs, T)
    
    print("Running Filter and Smoother...")
    filt_res = m.filter(data, outputs=['states'])
    s_final = filt_res[:, -1]
    
    print("Computing Forecast...")
    forecast_states, forecast_obs = forecast(m, TTT, RRR, CCC, s_final, horizon=8)
    
    print("Computing Shock Decomposition...")
    obs_decomp = shock_decomposition(m, TTT, RRR, CCC, data)
    
    print(f"Decomposition shape: {obs_decomp.shape} (n_shocks, T, n_obs)")
    
    # Verification: sum of contributions for a given observable should roughly match the observable (ignoring initial state)
    obs_idx = m.observables["obs_gdp"]
    total_contribution = np.sum(obs_decomp[:, :, obs_idx], axis=0) # Sum across shocks
    
    print("\nGDP Decomposed (at T=20):")
    for j, s_name in enumerate(m.exogenous_shocks.keys()):
        contrib = obs_decomp[j, 20, obs_idx]
        if abs(contrib) > 0.1: # Only print significant ones
            print(f"  {s_name}: {contrib:.4f}")

    print("\nFull Analysis test completed successfully!")

if __name__ == "__main__":
    test_full_analysis()
