import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# --- Physical Constants (SI Units) ---
k_B = 1.380649e-23     # Boltzmann constant (J/K)
T = 1e4                # Gas temperature (K)
G = 6.67430e-11        # Gravitational constant (m^3 / kg s^2)
m_p = 1.67262192e-27   # Proton mass (kg)
P_power = 60.0         # Field power per particle (W)
c = 299792458.0        # Speed of light (m/s)
pc_to_m = 3.086e16     # Conversion: parsecs to meters

# Derived dark mass constant
C_dark = P_power / (c**3)

# --- Density Functions ---
def rho(N):
    """Total effective mass density (kg/m^3) given number density N (m^-3)"""
    return m_p * N + C_dark * (N**(2/3))

def drho_dN(N):
    """Derivative of rho with respect to N"""
    return m_p + (2/3) * C_dark * (N**(-1/3))

# --- ODE System ---
def hydrostatic_ode(r, y):
    """
    Returns [dN/dr, d^2N/dr^2] for the SciPy solver.
    y[0] is N (density), y[1] is dN/dr (density gradient)
    """
    N, dN_dr = y
    
    # Catch to prevent invalid math if solver overshoots into negative density
    if N <= 1e-10:
        return [0, 0]
        
    term1 = - (2.0 / r) * dN_dr
    term2 = (drho_dN(N) / rho(N)) * (dN_dr**2)
    term3 = - (4 * np.pi * G / (k_B * T)) * (rho(N)**2)
    
    d2N_dr2 = term1 + term2 + term3
    
    return [dN_dr, d2N_dr2]

# --- Boundary Conditions & Setup ---
# Assume a central density of 0.5 particles/cm^3
N0_cm3 = 0.5  
N0 = N0_cm3 * 1e6  # Convert to particles/m^3

# Bypass the r=0 singularity by starting at epsilon (r0)
r0 = 1e-5 * pc_to_m 

# Calculate initial curvature using L'Hopital's limit for r->0
N_double_prime_0 = - (4 * np.pi * G / (3 * k_B * T)) * (rho(N0)**2)

# Taylor expansion for initial conditions at r0
N_initial = N0 + 0.5 * N_double_prime_0 * (r0**2)
dN_initial = N_double_prime_0 * r0

# Integration range: from r0 out to 5000 parsecs
r_span = (r0, 5000 * pc_to_m)
y0 = [N_initial, dN_initial]

# Stop the solver if density drops to near-zero (cloud edge)
def cloud_edge(r, y):
    return y[0] - 1.0  # Stop when N hits 1 particle/m^3
cloud_edge.terminal = True

# --- Execute Solver ---
# 'Radau' is used as it handles stiff equations well
solution = solve_ivp(hydrostatic_ode, r_span, y0, method='Radau', events=cloud_edge)

# --- Process & Plot Results ---
# Convert output back to astrophysics-friendly units for plotting
radius_pc = solution.t / pc_to_m
density_cm3 = solution.y[0] / 1e6

plt.figure(figsize=(8, 5))
plt.plot(radius_pc, density_cm3, color='indigo', linewidth=2)
plt.title("Gas Volume Density Profile (Custom Dark Matter Model)", fontsize=14)
plt.xlabel("Radius from Center (parsecs)", fontsize=12)
plt.ylabel(r"Gas Number Density ($\text{particles / cm}^3$)", fontsize=12)
plt.yscale("log")
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.show()