import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add src/python to path
sys.path.append(os.path.join(os.getcwd(), "src", "python"))

from dsge.models.m1002 import Model1002
from dsge.data import load_data, transform_data
from dsge.estimate import optimize_model
from dsge.samplers import MetropolisHastings

def run_real_estimation():
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        print("Error: FRED_API_KEY environment variable not set.")
        print("Please set it with: $env:FRED_API_KEY = 'your_key_here'")
        print("Or provide it now (it will not be saved):")
        api_key = input("FRED API Key: ").strip()
        if not api_key:
            print("No API key provided. Exiting.")
            return

    print("\n--- DSGE Model Estimation with Real FRED Data ---")
    
    # 1. Initialize Model
    print("Step 1: Initializing Model 1002...")
    m = Model1002()
    
    # 2. Fetch Data
    # Most DSGE models use a long historical sample, e.g., 1960 to recent
    start_date = "1960-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    print(f"Step 2: Fetching data from FRED ({start_date} to {end_date})...")
    levels = load_data(m, start_date, end_date, fred_api_key=api_key)
    
    if levels.empty or len(levels.columns) < 5:
        print("Error: Could not fetch sufficient data. Check your API key and internet connection.")
        return
        
    print(f"Fetched {len(levels)} quarters of raw data.")
    
    # 3. Transform Data
    print("Step 3: Transforming raw data into observables...")
    # Fill missing values with forward fill then backward fill for stability
    levels = levels.ffill().bfill()
    
    obs_data = transform_data(m, levels)
    
    # We want to drop early rows where almost nothing is available (start of series)
    # but we want to KEEP the dataset even if one variable (like TFP) is entirely missing
    # as the Kalman filter handles NaNs.
    
    # Drop rows where GDP is NaN (usually the limiting factor at the start)
    if "obs_gdp" in obs_data.columns:
        obs_data = obs_data.dropna(subset=["obs_gdp"]).reset_index(drop=True)
    
    print(f"Final dataset has {len(obs_data)} usable quarters.")
    print("Observables preview (last 5 quarters):")
    print(obs_data.tail())
    
    # Save for reference
    output_path = os.path.join("data", "fred_observables.csv")
    obs_data.to_csv(output_path, index=False)
    
    # 4. Estimation
    print("\nStep 4: Finding Posterior Mode (Maximum Likelihood + Priors)...")
    
    # Create the full data matrix (including NaNs for missing series)
    data_matrix = np.full((m.n_observables, len(obs_data)), np.nan)
    for i, key in enumerate(m.observables.keys()):
        if key in obs_data.columns:
            data_matrix[i, :] = obs_data[key].values
            
    # For a full run, we'd free many parameters. 
    # For this demo, we'll free a few core ones
    for p in m.parameters.values():
        p.fixed = True
    
    to_estimate = ["alp", "zeta_p", "rho", "psi1", "sigma_g", "sigma_b"]
    for k in to_estimate:
        m.parameters[k].fixed = False
    
    # Optimize
    res = optimize_model(m, data_matrix, method="L-BFGS-B")
    
    print("\nOptimization Results:")
    print(f"Success: {res.success}")
    print(f"Message: {res.message}")
    
    # Log results to outputs/results/
    log_file = os.path.join("outputs", "results", f"estimation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(log_file, "w") as f:
        f.write(f"Optimization Results:\nSuccess: {res.success}\nMessage: {res.message}\n")
        for i, k in enumerate(to_estimate):
            result_str = f"  {k}: {m.parameters[k].value:.4f} (Original raw: {res.x[i]:.4f})"
            print(result_str)
            f.write(result_str + "\n")
        
    # 5. Analysis (IRFs with est. parameters)
    print("\nStep 5: Generating IRFs with Optimized Parameters...")
    from dsge.utils.irf import compute_irf, irf_to_df
    TTT, RRR, CCC = m.solve()
    irf_states = compute_irf(m, TTT, RRR, CCC, m.exogenous_shocks["rm_sh"])
    df_irf = irf_to_df(m, irf_states, TTT=TTT, RRR=RRR, CCC=CCC)
    
    # Save IRF results
    irf_file = os.path.join("outputs", "results", "irf_results.csv")
    df_irf.to_csv(irf_file)
    
    print("Impact of Monetary Shock (1% rm_sh) on GDP and Inflation:")
    print(df_irf[["obs_gdp", "obs_corepce"]].head(1))
    
    print(f"\nDone! Data saved to '{output_path}' and results to '{log_file}'.")

if __name__ == "__main__":
    run_real_estimation()
