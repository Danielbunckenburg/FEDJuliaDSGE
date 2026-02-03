import numpy as np
import matplotlib.pyplot as plt
from dsge.models.m1002 import Model1002
from dsge.utils.irf import compute_irf, irf_to_df

def test_irfs():
    print("Initializing Model1002...")
    m = Model1002()
    
    print("Solving model...")
    TTT, RRR, CCC = m.solve()
    
    # 1. Monetery Policy Shock (rm_sh)
    shock_name = "rm_sh"
    shock_idx = m.exogenous_shocks[shock_name]
    
    print(f"Computing IRF for {shock_name}...")
    irf_states = compute_irf(m, TTT, RRR, CCC, shock_idx, horizon=20)
    df_obs = irf_to_df(m, irf_states, observables=True, TTT=TTT, RRR=RRR, CCC=CCC)
    
    # Plot some key observables
    keys = ["obs_gdp", "obs_corepce", "obs_nominalrate"]
    subset = df_obs[keys]
    
    print("\nIRF (Impact):")
    print(subset.iloc[0])
    
    print("\nIRF (4 quarters later):")
    print(subset.iloc[4])

    print("\nIRF First 5 quarters:")
    print(subset.head())

    # Check signs: A positive rm_sh (contractionary) should decrease GDP and Inflation
    assert subset["obs_gdp"].iloc[0] < 0 or subset["obs_gdp"].iloc[1] < 0, f"GDP should decrease, got {subset['obs_gdp'].iloc[0]}"
    assert subset["obs_nominalrate"].iloc[0] > 0, f"Nominal rate should increase, got {subset['obs_nominalrate'].iloc[0]}"
    
    print("\nIRF test passed successfully!")

if __name__ == "__main__":
    test_irfs()
