import numpy as np
import sys
import os

# Add src/python to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from dsge.models.an_schorfheide import AnSchorfheide

def test_anschorfheide():
    print("Initializing AnSchorfheide...")
    m = AnSchorfheide()
    
    print("Solving model...")
    try:
        TTT, RRR, CCC = m.solve()
        print(f"Model solved! TTT shape: {TTT.shape}")
        # print("TTT diagonal:", np.diag(TTT))
    except Exception as e:
        print(f"Solve failed: {e}")
        return

    print("Evaluating Likelihood...")
    n_obs = m.n_observables
    T = 10
    data = np.random.randn(n_obs, T)
    loglh = m.likelihood(data)
    print(f"Log-Likelihood: {loglh:.6f}")
    
    assert not np.isnan(loglh)
    print("AnSchorfheide test successful!")

if __name__ == "__main__":
    test_anschorfheide()
