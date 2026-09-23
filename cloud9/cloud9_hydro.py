import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# --- Physical Constants (SI Units) ---
k_B = 1.380649e-23     # J/K
T = 3e5                # Gas temperature (K)
G = 6.67430e-11        # m^3 / (kg s^2)
m_p = 1.67262192e-27   # Proton mass (kg)
P_power = 60      # Field power (W)
c = 299792458.0        # Speed of light (m/s)

# --- Conversion Factors ---
pc_to_m = 3.086e16
kpc_to_m = 3.086e19
kg_to_Msun = 1.0 / 1.98847e30
kg_to_amu = 1.0 / 1.66053906660e-27  # Convert kg to atomic mass units
kg_m3_to_amu_cm3 = kg_to_amu / 1e6   # Convert kg/m^3 to amu/cm^3

C_dark = P_power / (c**3)

# --- Density Thresholds ---
min_density_cm3 = 0.0000002
N_min = min_density_cm3 * 1e6  # Convert to particles/m^3

# --- Density Functions (with Cutoff) ---
def rho(N):
    """Total mass density (kg/m^3). DM breaks down below N_min."""
    if N < N_min:
        N = N_min
        #return m_p * N
    return m_p * N + C_dark * (N**(2/3))

def drho_dN(N):
    """Derivative of total mass density w.r.t N."""
    if N < N_min:
        return 0
    return m_p + (2/3) * C_dark * (N**(-1/3))

# Vectorized version of rho for processing the final arrays
def rho_array(N_arr):
    res = np.zeros_like(N_arr)
    mask = N_arr < N_min
    res[mask] = m_p * N_arr[mask]
    res[~mask] = m_p * N_arr[~mask] + C_dark * (N_arr[~mask]**(2/3))
    return res

# --- ODE System ---
def hydrostatic_ode(r, y):
    N, dN_dr, M_gas, M_tot = y
    
    if N <= N_min:
        N = N_min
        dN_dr = 0
 
    # if N <= 1e-14:
    #     return [0, 0, 0, 0]
        
    term1 = - (2.0 / r) * dN_dr
    term2 = (drho_dN(N) / rho(N)) * (dN_dr**2)
    term3 = - (4 * np.pi * G / (k_B * T)) * (rho(N)**2)
    
    d2N_dr2 = term1 + term2 + term3
    
    # dM/dr for Enclosed Mass
    dM_gas_dr = 4 * np.pi * (r**2) * m_p * N
    dM_tot_dr = 4 * np.pi * (r**2) * rho(N)
    
    return [dN_dr, d2N_dr2, dM_gas_dr, dM_tot_dr]

# --- Boundary Conditions ---
N0_cm3 = 0.5  
N0 = N0_cm3 * 1e6  # particles/m^3

r0 = 1e-10 * pc_to_m 
N_double_prime_0 = - (4 * np.pi * G / (3 * k_B * T)) * (rho(N0)**2)

N_initial = N0 + 0.5 * N_double_prime_0 * (r0**2)
dN_initial = N_double_prime_0 * r0
M_gas_initial = (4/3) * np.pi * (r0**3) * m_p * N_initial
M_tot_initial = (4/3) * np.pi * (r0**3) * rho(N_initial)

y0 = [N_initial, dN_initial, M_gas_initial, M_tot_initial]

# Max integration distance set to 50 kpc
r_max_m = 10000 * pc_to_m  
r_span = (r0, r_max_m)

def cloud_edge(r, y):
    return y[0] - 1e-6  # Stop if N drops to near absolute zero
cloud_edge.terminal = True

# --- Execute Solver ---
solution = solve_ivp(hydrostatic_ode, r_span, y0, method='Radau', events=cloud_edge)

# Extract results
r_arr = solution.t
N_arr = solution.y[0]
M_gas_arr = solution.y[2]
M_tot_arr = solution.y[3]

# --- Convert Units for Plotting ---
r_kpc = r_arr / kpc_to_m
M_gas_Msun = M_gas_arr * kg_to_Msun
M_tot_Msun = M_tot_arr * kg_to_Msun

# Convert density array to amu / cm^3
rho_tot_plot = rho_array(N_arr) * kg_m3_to_amu_cm3
rho_gas_plot = (m_p * N_arr) * kg_m3_to_amu_cm3

# Print Final Masses
print(f"Integration ended at: {r_kpc[-1]:.2f} kpc")
print(f"Total Cloud Mass: {M_tot_Msun[-1]:.2e} M_sun")
print(f"Total Gas Mass:   {M_gas_Msun[-1]:.2e} M_sun")

# Auto-scaling logic (edge of integration)
x_max = min(10, r_kpc[-1]*1.2)

# --- Plotting ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Graph 1: Density Profile
ax1.plot(r_kpc, rho_tot_plot, label='Total Mass Density', color='indigo', linewidth=2)
ax1.plot(r_kpc, rho_gas_plot, label='Gas Mass Density', color='orange', linewidth=2, linestyle='--')
ax1.axhline(min_density_cm3 * (m_p * kg_to_amu), color='red', linestyle=':', label='DM Cutoff Threshold')

ax1.set_title(f"Cloud 9 $T={T:.2e}$K, Power = {P_power} W, N_i = {N0_cm3} particles/cm$^3$")
ax1.set_xlabel("Radius (kpc)")
ax1.set_ylabel(r"Mass Density (amu / cm$^3$) (hydrostatic EQ)")
ax1.set_yscale("log")
ax1.set_xlim(0, 2)
ax1.grid(True, which="both", ls="--", alpha=0.5)
ax1.legend()

# Graph 2: Enclosed Mass
ax2.plot(r_kpc[30:], M_tot_Msun[30:], label='Enclosed Total Mass', color='indigo', linewidth=2)
ax2.plot(r_kpc[30:], M_gas_Msun[30:], label='Enclosed Gas Mass', color='orange', linewidth=2, linestyle='--')

ax2.set_title("Cumulative Enclosed Mass")
ax2.set_xlabel("Radius (kpc)")
ax2.set_ylabel(r"Enclosed Mass ($M_\odot$)")
ax2.set_yscale("log")
ax2.set_xlim(0.01, 10)
ax2.set_xscale("log")
ax2.grid(True, ls="--", alpha=0.5)
ax2.legend()

plt.tight_layout()
plt.show()

