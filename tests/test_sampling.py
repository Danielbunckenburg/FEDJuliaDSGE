import numpy as np
from dsge.models.m1002 import Model1002
from dsge.samplers import MetropolisHastings
from dsge.estimate import optimize_model

def test_sampling():
    print("Initializing Model1002...")
    m = Model1002()
    
    print("Generating simulated data...")
    n_obs = m.n_observables
    T = 40
    data = np.random.randn(n_obs, T) # Dummy data for testing logic
    
    # Fix most parameters
    for p in m.parameters.values():
        p.fixed = True
    
    # Free up 3 parameters for a small joint estimation
    m.parameters["alp"].fixed = False
    m.parameters["zeta_p"].fixed = False
    m.parameters["sigma_g"].fixed = False
    
    print("Finding mode (L-BFGS-B)...")
    res = optimize_model(m, data, method="L-BFGS-B")
    print(f"Mode found: {res.x}")
    
    # Use the inverse Hessian (if available) or a simplified proposal cov
    # For this test, we'll just use identity scaled
    sampler = MetropolisHastings(m, data, scale=0.1)
    
    n_samples = 100
    burn_in = 20
    
    chain, log_posts = sampler.sample(n_samples, burn_in=burn_in)
    
    print("\nPosterior Means:")
    for i, key in enumerate(sampler.param_keys):
        mean_val = np.mean(chain[:, i])
        std_val = np.std(chain[:, i])
        print(f"{key}: {mean_val:.4f} +/- {std_val:.4f}")
    
    print("\nSampling test completed successfully.")

if __name__ == "__main__":
    test_sampling()
