import numpy as np
from dsge.models.m1002 import Model1002
from dsge.estimate import optimize_model
from dsge.priors import Normal, Gamma, Beta, InverseGamma

def test_optimization():
    print("Initializing Model1002...")
    m = Model1002()
    
    # Add dummy priors to a few parameters for testing
    # In a real scenario, all parameters would have priors defined in m1002.py
    m.parameters["alp"].prior = Normal(0.3, 0.05)
    m.parameters["bet"].prior = Gamma(2.0, 0.1) # Just a guess
    
    print("Generating simulated data...")
    # Simulate data at the true parameters
    TTT, RRR, CCC = m.solve()
    s0, P0 = m.init_stationary_states(TTT, RRR, m.measurement(TTT, RRR, CCC)[2]) # QQ from measurement
    
    # We need to simulate states properly
    # Simple simulation for testing
    n_obs = m.n_observables
    T = 20
    
    # Just random noise for data is enough to run the optimizer code, 
    # though it won't converge to "truth"
    data = np.random.randn(n_obs, T)
    
    print("Initial Log-Posterior:", m.posterior(data))
    
    print("Running optimization (short run)...")
    # Fix most parameters to make it fast
    for p in m.parameters.values():
        p.fixed = True
    
    # Free up a couple
    m.parameters["alp"].fixed = False
    m.parameters["zeta_p"].fixed = False
    
    res = optimize_model(m, data, method="L-BFGS-B")
    
    print("Optimization success:", res.success)
    print("Final Log-Posterior:", -res.fun)
    print("Optimized alpha:", m.parameters["alp"].value)
    print("Optimized zeta_p:", m.parameters["zeta_p"].value)

if __name__ == "__main__":
    test_optimization()
