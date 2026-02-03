import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dsge.models.m1002 import Model1002
from dsge.utils.irf import compute_irf, irf_to_df

def run_experiments():
    output_dir = "outputs/plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Initializing Model...")
    m = Model1002()
    
    # Define interesting observables to plot
    plot_obs = ["obs_gdp", "obs_corepce", "obs_nominalrate"]
    labels = {"obs_gdp": "Output Growth", "obs_corepce": "Inflation", "obs_nominalrate": "Interest Rate"}

    # --- Experiment 1: Baseline IRFs ---
    print("Running Experiment 1: Baseline IRFs...")
    TTT, RRR, CCC = m.solve()
    
    shocks = [("rm_sh", "Monetary Policy Shock"), ("ztil_sh", "Technology Shock")]
    
    for shock_id, shock_name in shocks:
        shock_idx = m.exogenous_shocks[shock_id]
        irf_matrix = compute_irf(m, TTT, RRR, CCC, shock_idx, horizon=20)
        df_irf = irf_to_df(m, irf_matrix, observables=True, TTT=TTT, RRR=RRR, CCC=CCC)
        
        plt.figure(figsize=(12, 4))
        for i, obs in enumerate(plot_obs):
            plt.subplot(1, 3, i+1)
            plt.plot(df_irf[obs])
            plt.title(f"{labels[obs]}")
            plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"irf_{shock_id}.png"))
        plt.close()

    # --- Experiment 2: Taylor Rule Sensitivity (psi1) ---
    print("Running Experiment 2: Taylor Rule Sensitivity...")
    shock_id = "rm_sh"
    shock_idx = m.exogenous_shocks[shock_id]
    
    psi1_values = [1.5, 2.5]
    plt.figure(figsize=(12, 4))
    
    for p_val in psi1_values:
        m.parameters["psi1"].value = p_val
        TTT, RRR, CCC = m.solve()
        irf_matrix = compute_irf(m, TTT, RRR, CCC, shock_idx, horizon=20)
        df_irf = irf_to_df(m, irf_matrix, observables=True, TTT=TTT, RRR=RRR, CCC=CCC)
        
        for i, obs in enumerate(plot_obs):
            plt.subplot(1, 3, i+1)
            plt.plot(df_irf[obs], label=f"psi1={p_val}")
            plt.title(f"{labels[obs]}")
            plt.grid(True, linestyle='--', alpha=0.7)
            if i == 0: plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sensitivity_psi1.png"))
    plt.close()

    # --- Experiment 3: Price Stickiness Sensitivity (zeta_p) ---
    print("Running Experiment 3: Price Stickiness Sensitivity...")
    # Reset psi1 to default
    m.parameters["psi1"].value = 1.3679 
    
    zeta_values = [0.5, 0.8] # 0.5 is flexible-ish, 0.8 is sticky
    plt.figure(figsize=(12, 4))
    
    for z_val in zeta_values:
        m.parameters["zeta_p"].value = z_val
        TTT, RRR, CCC = m.solve()
        irf_matrix = compute_irf(m, TTT, RRR, CCC, shock_idx, horizon=20)
        df_irf = irf_to_df(m, irf_matrix, observables=True, TTT=TTT, RRR=RRR, CCC=CCC)
        
        for i, obs in enumerate(plot_obs):
            plt.subplot(1, 3, i+1)
            plt.plot(df_irf[obs], label=f"zeta_p={z_val}")
            plt.title(f"{labels[obs]}")
            plt.grid(True, linestyle='--', alpha=0.7)
            if i == 0: plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sensitivity_zeta_p.png"))
    plt.close()

    print(f"All experiments complete. Plots saved to {output_dir}")

if __name__ == "__main__":
    run_experiments()
