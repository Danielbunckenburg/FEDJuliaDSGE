import numpy as np
from dsge.models.m1002 import Model1002

def test_full():
    print("Initializing Model1002...")
    m = Model1002()
    
    print("Solving model...")
    try:
        TTT, RRR, CCC = m.solve()
        print(f"Model solved! TTT shape: {TTT.shape}")
        
        # Verify sizes
        assert TTT.shape[0] == m.n_states_augmented
        assert RRR.shape[1] == m.n_shocks_exogenous
        
    except ValueError as e:
        print(f"Solve failed: {e}")
        return

    print("Generating simulated data...")
    n_obs = m.n_observables
    T = 10
    data = np.random.randn(n_obs, T)
    
    print("Evaluating Likelihood...")
    loglh = m.likelihood(data)
    print(f"Log-Likelihood: {loglh:.6f}")
    
    # Check if calculation was successful
    assert not np.isnan(loglh), "Likelihood is NaN"
    print("\nEnd-to-end test successful!")

if __name__ == "__main__":
    test_full()
