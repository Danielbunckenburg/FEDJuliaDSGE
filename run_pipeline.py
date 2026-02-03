import os
import numpy as np
from dsge.models.m1002 import Model1002
from dsge.utils.data_loader import load_csv_data, create_data_template

def run_pipeline():
    # 1. Initialize
    print("Step 1: Initializing Model...")
    m = Model1002()
    
    # 2. Data
    print("\nStep 2: Loading Data...")
    # For demonstration, we create a template and fill it with noise
    data_dir = "data"
    data_file = os.path.join(data_dir, "data_sample.csv")
    if not os.path.exists(data_file):
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        create_data_template(m, data_file, T=40)
    
    # (Simulated data for now as we don't have real FRED keys yet)
    data = np.random.randn(m.n_observables, 40)
    print(f"Data Loaded: {data.shape}")
    
    # 3. Estimation (Mode Finding)
    print("\nStep 3: Finding Posterior Mode (Simplified)...")
    # Fix most parameters to make it fast for demo
    for p in m.parameters.values(): p.fixed = True
    m.parameters["alp"].fixed = False
    m.parameters["zeta_p"].fixed = False
    
    from dsge.estimate import optimize_model
    res = optimize_model(m, data, method="L-BFGS-B")
    print(f"Optimal Alpha: {m.parameters['alp'].value:.4f}")
    
    # 4. Analysis
    print("\nStep 4: Running Diagnostics...")
    TTT, RRR, CCC = m.solve()
    
    # IRFs
    from dsge.utils.irf import compute_irf
    irf = compute_irf(m, TTT, RRR, CCC, m.exogenous_shocks["rm_sh"])
    print(f"IRF Impact on GDP: {irf[0, m.endogenous_states['y_t']]:.4f}")
    
    # Forecast
    from dsge.forecast import forecast
    filt_res = m.filter(data, outputs=['states'])
    s_final = filt_res[:, -1]
    _, f_obs = forecast(m, TTT, RRR, CCC, s_final, horizon=12)
    print(f"Forecasted GDP (next qtr): {f_obs[0, m.observables['obs_gdp']]:.4f}")

    print("\nPipeline complete!")

if __name__ == "__main__":
    run_pipeline()
