from dsge.models.m1002 import Model1002
import numpy as np

def verify_ss():
    print("Initializing Model1002...")
    m = Model1002()
    
    print("\nSteady State Results:")
    print(f"z_star:   {m.z_star:.6f}")
    print(f"rstar:    {m.rstar:.6f}")
    print(f"Rstarn:   {m.Rstarn:.6f}")
    print(f"wstar:    {m.wstar:.6f}")
    print(f"Lstar:    {m.Lstar:.6f}")
    print(f"kstar:    {m.kstar:.6f}")
    print(f"ystar:    {m.ystar:.6f}")
    print(f"cstar:    {m.cstar:.6f}")
    print(f"istar:    {m.istar:.6f}")
    print(f"sigma_omega_star: {m.sigma_omega_star:.6f}")
    
    # Check if calculation was successful
    assert not np.isnan(m.z_star), "z_star is NaN"
    assert m.ystar > 0, "ystar should be positive"
    
    print("\nVerification successful (internal consistency)!")

if __name__ == "__main__":
    verify_ss()
