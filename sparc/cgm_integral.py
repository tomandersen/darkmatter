from scipy.ndimage import morphology
import numpy as np
from scipy.integrate import quad

m_p = 1.67262e-27      

# SI units assumed. (I do no units here!)
def density_integrand(r, gas_only, P, c, rc1, n0_1, exponent1, rc2, n0_2, exponent2, min_gas_for_dm):
    # first, I need to calulate the density of the gas particles at r:
    # rho(r) = rho_0 [ 1 + (r/rc)^2 ]^(-3beta/2)
    gas_density_at_r = 0
    if n0_1 > 0:
      gas_density_at_r += n0_1*(1.0 + (r / rc1)**2)**(exponent1)
    if n0_2 > 0:
      gas_density_at_r += n0_2*(1.0 + (r / rc2)**2)**(exponent2)
    if gas_only:
      return 4*np.pi*r**2*gas_density_at_r

    if gas_density_at_r < min_gas_for_dm:
        gas_density_at_r = 0.0
 
    # ok now we have the gas density, get the dm density from our formula
    dm_particle_density = (P/c**3)*(gas_density_at_r)**(2.0/3.0)/m_p

    # ok now we have the dm particle density, get the dm mass from that (need to multiply by volume)
    dm_mass_shell = dm_particle_density * (4*np.pi*r**2)
    return dm_mass_shell

def evaluate_dm_mass_integral(R1, R2, gas_only, P, c, rc1, n0_1, exponent1, rc2, n0_2, exponent2, min_gas_for_dm):
    """
    Evaluates the integral of our halo(r) from R1 to R2.
    
    Parameters:
    R2    : Upper limit of integration (radius)
    P    : Power in Watts
    c    : Speed of light metre/sec
    n0   : Central density parameter, particles per metre^3
    rc   : Core radius - in metres
    exponent : full (negative) exponent - if you have beta, pass -3/2*beta. 
    """
    # Calculate the constant prefactor outside the integral
    # This prevents potential floating-point underflow/overflow inside the integration loop
    
    # Evaluate the definite integral from 0 to R
    # quad returns a tuple: (integral_value, absolute_error_estimate)
    integral_val, abs_error = quad(density_integrand, R1, R2, args=(gas_only, P, c, rc1, n0_1, exponent1, rc2, n0_2, exponent2, min_gas_for_dm))
    #print(f"integral_val (unitless)     : {integral_val:.6e}")
    return integral_val, abs_error

if __name__ == "__main__":
    # --- Example Usage ---
    # Define physical parameters to test the script - using NGC0289
    R_200 = 500000.0       # Integration limit
    P_val = 10       # Power, Watts
    c_val = 3e8       # Speed of light
    n0_val = 1      # Central number density
    rc_val = 10.0     # Core radius
    beta_val = 3.0*(2.0/3.0) # 2/3 is the cannonical beta, but I am adding mass to galaxy cores...
    
    R1 = 0
    R2 = 100000000
    gas_only=False
    P = 10.0
    c = 3e8
    rc1=100000.0
    n0_1=1
    exponent1=beta_val
    rc2=100000.0
    n0_2=1
    exponent2=beta_val
    min_gas_for_dm = 0.0000000001

    result, error = evaluate_dm_mass_integral(R1, R2, gas_only, P, c, rc1, n0_1, exponent1, rc2, n0_2, exponent2, min_gas_for_dm)
    
