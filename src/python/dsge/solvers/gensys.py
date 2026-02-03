import numpy as np
from scipy.linalg import qz, ordqz

def gensys(g0, g1, c, psi, pi, div=1.01):
    """
    Python implementation of Sims (2002) gensys solver.
    g0*y(t) = g1*y(t-1) + c + psi*z(t) + pi*eta(t)
    """
    n = g0.shape[0]
    
    # QZ Decomposition (Generalized Schur)
    # Julia uses complex schur by default for gensys to match Matlab
    S, T, Q, Z = qz(g0, g1, output='complex')
    
    # Stability criterion
    # select[i] = !(abs(b[i, i]) > div * abs(a[i, i]))
    def select_stable(alpha, beta):
        return np.abs(beta) <= div * np.abs(alpha)

    # Reorder
    S, T, alpha, beta, Q, Z = ordqz(g0, g1, sort=select_stable, output='complex')
    
    # Let n = number of variables
    # nunstab = number of unstable roots (where np.abs(beta) > div * np.abs(alpha))
    nunstab = n - np.sum(select_stable(np.diag(S), np.diag(T)))
    nstable = n - nunstab
    
    # print(f"DEBUG: n={n}, nstable={nstable}, nunstab={nunstab}")
    
    # Existence and Uniqueness check (simplified)
    eu = [1, 1]
    
    try:
        if nstable > 0:
            S11 = S[:nstable, :nstable]
            T11 = T[:nstable, :nstable]
            cond = np.linalg.cond(S11)
            # print(f"DEBUG: S11 cond={cond:.2e}")
            
            # Use lstsq for robustness
            X, residuals, rank, s = np.linalg.lstsq(S11, T11, rcond=None)
            G1 = Z[:, :nstable] @ X @ Z[:, :nstable].conj().T
        else:
            G1 = np.zeros((n, n))
        
        # impact = Z11 * S11^-1 * Q1' * psi
        if nstable > 0:
            Q1psi = Q.conj().T[:nstable, :] @ psi
            impact_part, _, _, _ = np.linalg.lstsq(S[:nstable, :nstable], Q1psi, rcond=None)
            impact = Z[:, :nstable] @ impact_part
            # print(f"DEBUG: gensys psi={psi.shape}, impact_part={impact_part.shape}, impact={impact.shape}")
        else:
            impact = np.zeros((n, psi.shape[1]))
            
        # C = Z11 * (S11 - T11)^-1 * Q1' * c
        if nstable > 0:
            Q1c = Q.conj().T[:nstable, :] @ c
            C_part, _, _, _ = np.linalg.lstsq(S[:nstable, :nstable] - T[:nstable, :nstable], Q1c, rcond=None)
            C = Z[:, :nstable] @ C_part
        else:
            C = np.zeros(n)
        
        return np.real(G1), np.real(C), np.real(impact), eu
    except Exception as e:
        print(f"Gensys Error: {e}")
        return None, None, None, [-3, -3]
