import numpy as np
from scipy.integrate import quad

def density_integrand(r, rc, beta):
    """
    The r-dependent part of the density profile.
    """
    return 4*np.pi*r**2*(1.0 + (r / rc)**2)**(-beta)

def evaluate_mass_integral(R, P, c, n0, rc, beta):
    """
    Evaluates the integral of rho_dm(r) from 0 to R.
    
    Parameters:
    R    : Upper limit of integration (radius)
    P    : Pressure parameter
    c    : Speed of light (or relevant velocity scale)
    n0   : Central density parameter
    rc   : Core radius
    beta : Slope parameter
    """
    # Calculate the constant prefactor outside the integral
    # This prevents potential floating-point underflow/overflow inside the integration loop
    prefactor = (P / c**3) * (n0**(2.0/3.0))
    
    # Evaluate the definite integral from 0 to R
    # quad returns a tuple: (integral_value, absolute_error_estimate)
    integral_val, abs_error = quad(density_integrand, 0, R, args=(rc, beta))
    print(f"integral_val (unitless)     : {integral_val:.6e}")

    print(f"P     : {P:.6e}")
    print(f"c^3     : {c**3:.6e}")
    print(f"n0^(2/3)     : {n0**(2.0/3.0):.6e}")
    print(f"prefactor     : {prefactor:.6e}")
    
    # Scale the result and error by the prefactor
    total_value = prefactor * integral_val
    total_error = prefactor * abs_error
    
    print(f"Integration limit R : {R}")
    print(f"Integral result     : {total_value:.6e}")
    print(f"Estimated error     : {total_error:.6e}")

    return total_value, total_error

if __name__ == "__main__":
    # --- Example Usage ---
    # Define arbitrary physical parameters to test the script
    R_max = 500000.0       # Integration limit
    P_val = 10       # Power, Watts
    c_val = 3e8       # Speed of light
    n0_val = 1      # Central number density
    rc_val = 10.0     # Core radius
    beta_val = 3.0*(2.0/3.0) # 2/3 is the cannonical beta, but I am adding mass to galaxy cores...
    
    result, error = evaluate_mass_integral(R_max, P_val, c_val, n0_val, rc_val, beta_val)
    
