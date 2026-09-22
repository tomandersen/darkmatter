import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# --- Physical Constants (SI Units) ---
k_B = 1.380649e-23     # J/K
T = 1e4                # Gas temperature (K)
G = 6.67430e-11        # m^3 / (kg s^2)
m_p = 1.67262192e-27   # Proton mass (kg)
P_power = 60.0         # Field power (W)
c = 299792458.0        # Speed of light (m/s)

# --- Conversion Factors ---
pc_to_m = 3.086e16
kpc_to_m = 3.086e19
kg_to_Msun = 1.0 / 1.98847e30
m3_to_pc3 = (3.086e16)**3

C_dark = P_power / (c**3)

# --- Density Functions ---
def rho(N):
    """Total mass density (kg/m^3)"""
    return m_p * N + C_dark * (N**(2/3))

def drho_dN(N):
    """Derivative of total mass density w.r.t N"""
    return m_p + (2/3) * C_dark * (N**(-1/3))

# --- ODE System ---
def hydrostatic_ode(r, y):
    """
    y = [N, dN/dr, M_gas, M_tot]
    """
    N, dN_dr, M_gas, M_tot = y
    
    # Stop integration math errors if density hits 0
    if N <= 1e-10:
        return [0, 0, 0, 0]
        
    term1 = - (2.0 / r) * dN_dr
    term2 = (drho_dN(N) / rho(N)) * (dN_dr**2)
    term3 = - (4 * np.pi * G / (k_B * T)) * (rho(N)**2)
    
    d2N_dr2 = term1 + term2 + term3
    
    # dM/dr for Enclosed Mass
    dM_gas_dr = 4 * np.pi * (r**2) * m_p * N
    dM_tot_dr = 4 * np.pi * (r**2) * rho(N)
    
    return [dN_dr, d2N_dr2, dM_gas_dr, dM_tot_dr]

# --- Boundary Conditions ---
N0_cm3 = 100  
N0 = N0_cm3 * 1e6  # particles/m^3

r0 = 1e-5 * pc_to_m 
N_double_prime_0 = - (4 * np.pi * G / (3 * k_B * T)) * (rho(N0)**2)

# Initial values at tiny r0
N_initial = N0 + 0.5 * N_double_prime_0 * (r0**2)
dN_initial = N_double_prime_0 * r0
M_gas_initial = (4/3) * np.pi * (r0**3) * m_p * N_initial
M_tot_initial = (4/3) * np.pi * (r0**3) * rho(N_initial)

y0 = [N_initial, dN_initial, M_gas_initial, M_tot_initial]
r_max_m = 30000 * pc_to_m  # 30 kpc
r_span = (r0, r_max_m)

# Stop the solver exactly at the cloud edge (N -> 0)
def cloud_edge(r, y):
    return y[0] - 1e-5  # Stop when N drops to near zero
cloud_edge.terminal = True

# --- Execute Solver ---
solution = solve_ivp(hydrostatic_ode, r_span, y0, method='Radau', events=cloud_edge)

# Extract results
r_arr = solution.t
N_arr = solution.y[0]
M_gas_arr = solution.y[2]
M_tot_arr = solution.y[3]

# --- Pad Data out to 30 kpc (if cloud terminates early) ---
if r_arr[-1] < r_max_m:
    # Create empty space out to 30 kpc
    pad_r = np.linspace(r_arr[-1], r_max_m, 200)[1:]
    
    r_arr = np.concatenate((r_arr, pad_r))
    # Fill density with NaN so the graph line stops cleanly
    N_arr = np.concatenate((N_arr, np.full_like(pad_r, np.nan)))
    # Mass remains flat (constant) in empty space
    M_gas_arr = np.concatenate((M_gas_arr, np.full_like(pad_r, M_gas_arr[-1])))
    M_tot_arr = np.concatenate((M_tot_arr, np.full_like(pad_r, M_tot_arr[-1])))

# --- Convert Units for Plotting ---
r_kpc = r_arr / kpc_to_m
M_gas_Msun = M_gas_arr * kg_to_Msun
M_tot_Msun = M_tot_arr * kg_to_Msun

# Convert Number Density to Mass Density (M_sun / pc^3) for plotting
# Using np.where to avoid invalid math warnings on NaN values
rho_tot_plot = np.where(np.isnan(N_arr), np.nan, rho(N_arr) * m3_to_pc3 * kg_to_Msun)
rho_gas_plot = np.where(np.isnan(N_arr), np.nan, m_p * N_arr * m3_to_pc3 * kg_to_Msun)

# --- Plotting ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Graph 1: Density Profile
ax1.plot(r_kpc, rho_tot_plot, label='Total Mass Density', color='indigo', linewidth=2)
ax1.plot(r_kpc, rho_gas_plot, label='Gas Mass Density', color='orange', linewidth=2, linestyle='--')
ax1.set_title("Volume Mass Density Profile")
ax1.set_xlabel("Radius (kpc)")
ax1.set_ylabel(r"Mass Density ($M_\odot$ / pc$^3$)")
ax1.set_yscale("log")
ax1.set_xlim(0, 0.2)
ax1.grid(True, which="both", ls="--", alpha=0.5)
ax1.legend()

# Graph 2: Enclosed Mass
ax2.plot(r_kpc, M_tot_Msun, label='Enclosed Total Mass', color='indigo', linewidth=2)
ax2.plot(r_kpc, M_gas_Msun, label='Enclosed Gas Mass', color='orange', linewidth=2, linestyle='--')
ax2.set_title("Cumulative Enclosed Mass")
ax2.set_xlabel("Radius (kpc)")
ax2.set_ylabel(r"Enclosed Mass ($M_\odot$)")
ax2.set_xlim(0, 0.2)
ax2.grid(True, ls="--", alpha=0.5)
ax2.legend()

plt.tight_layout()
plt.show()

# Print Final Masses
print(f"Total Cloud Mass: {M_tot_Msun[-1]:.2e} M_sun")
print(f"Total Gas Mass:   {M_gas_Msun[-1]:.2e} M_sun")