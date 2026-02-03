import numpy as np
from scipy.stats import norm
from numpy.linalg import matrix_power

def ω_fn(z, sigma):
    return np.exp(sigma * z - sigma**2 / 2)

def G_fn(z, sigma):
    return norm.cdf(z - sigma)

def Γ_fn(z, sigma):
    return ω_fn(z, sigma) * (1 - norm.cdf(z)) + norm.cdf(z - sigma)

def dG_dω_fn(z, sigma):
    return norm.pdf(z) / sigma

def d2G_dω2_fn(z, sigma):
    return -z * norm.pdf(z) / (ω_fn(z, sigma) * sigma**2)

def dΓ_dω_fn(z):
    return 1 - norm.cdf(z)

def d2Γ_dω2_fn(z, sigma):
    return -norm.pdf(z) / (ω_fn(z, sigma) * sigma)

def μ_fn(z, sigma, spr):
    num = (1 - 1/spr)
    den = (dG_dω_fn(z, sigma) / dΓ_dω_fn(z) * (1 - Γ_fn(z, sigma)) + G_fn(z, sigma))
    return num / den

def nk_fn(z, sigma, spr):
    return 1 - (Γ_fn(z, sigma) - μ_fn(z, sigma, spr) * G_fn(z, sigma)) * spr

def ζ_bω_fn(z, sigma, spr):
    nk = nk_fn(z, sigma, spr)
    mu = μ_fn(z, sigma, spr)
    omega = ω_fn(z, sigma)
    gamma = Γ_fn(z, sigma)
    g = G_fn(z, sigma)
    dg = dG_dω_fn(z, sigma)
    dgamma = dΓ_dω_fn(z)
    d2g = d2G_dω2_fn(z, sigma)
    d2gamma = d2Γ_dω2_fn(z, sigma)
    
    num = omega * mu * nk * (d2gamma * dg - d2g * dgamma)
    den = (dgamma - mu * dg)**2 * spr * (1 - gamma + dgamma * (gamma - mu * g) / (dgamma - mu * dg))
    return num / den

def ζ_zω_fn(z, sigma, spr):
    mu = μ_fn(z, sigma, spr)
    return ω_fn(z, sigma) * (dΓ_dω_fn(z) - mu * dG_dω_fn(z, sigma)) / (Γ_fn(z, sigma) - mu * G_fn(z, sigma))

def ζ_spb_fn(z, sigma, spr):
    zetaratio = ζ_bω_fn(z, sigma, spr) / ζ_zω_fn(z, sigma, spr)
    nk = nk_fn(z, sigma, spr)
    return -zetaratio / (1 - zetaratio) * nk / (1 - nk)

def k_periods_ahead_expectations(TTT, CCC, k):
    """
    Computes E_t[s_{t+k}] = TTT^k * s_t + (I + TTT + ... + TTT^{k-1}) * CCC
    Simplified for steady CCC.
    """
    Tk = matrix_power(TTT, k)
    # Sum of geometric series: (I - TTT^k) * (I - TTT)^-1
    if np.all(CCC == 0):
        return Tk, np.zeros_like(CCC)
    
    n = TTT.shape[0]
    I = np.eye(n)
    try:
        S = np.linalg.solve(I - TTT, I - Tk)
        Ck = S @ CCC
    except np.linalg.LinAlgError:
        # Fallback to manual sum if I-TTT is singular
        Ck = np.zeros_like(CCC)
        current_T = I
        for _ in range(k):
            Ck += current_T @ CCC
            current_T = current_T @ TTT
            
    return Tk, Ck

def k_periods_ahead_expected_sums(TTT, CCC, k):
    """
    Computes sum_{i=1}^k E_t[s_{t+i}]
    """
    n = TTT.shape[0]
    I = np.eye(n)
    
    # sum_{i=1}^k T^i = T * (I - T^k) * (I - T)^-1
    try:
        inv_I_T = np.linalg.solve(I - TTT, I)
        Tk = matrix_power(TTT, k)
        T_sum = TTT @ (I - Tk) @ inv_I_T
        
        # This is more complex for CCC, but usually CCC is small or zero in these models
        # For now, let's just do a loop if necessary, but most DSGEs use CCC=0 for this.
        C_sum = np.zeros_like(CCC)
        if not np.all(CCC == 0):
            current_Ck_sum = np.zeros_like(CCC)
            current_Tk = I
            for i in range(1, k + 1):
                Tk_i, Ck_i = k_periods_ahead_expectations(TTT, CCC, i)
                C_sum += Ck_i
    except:
        # Manual loop fallback
        T_sum = np.zeros_like(TTT)
        C_sum = np.zeros_like(CCC)
        current_T = I
        for i in range(1, k + 1):
            current_T = current_T @ TTT
            T_sum += current_T
            Tk_i, Ck_i = k_periods_ahead_expectations(TTT, CCC, i)
            C_sum += Ck_i
            
    return T_sum, C_sum

# Additional derivatives for elasticities
def dG_dσ_fn(z, sigma):
    return -z * norm.pdf(z - sigma) / sigma

def d2G_dωdσ_fn(z, sigma):
    return -norm.pdf(z) * (1 - z * (z - sigma)) / sigma**2

def dΓ_dσ_fn(z, sigma):
    return -norm.pdf(z - sigma)

def d2Γ_dωdσ_fn(z, sigma):
    return (z / sigma - 1) * norm.pdf(z)
