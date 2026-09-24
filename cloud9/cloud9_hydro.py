from matplotlib import mathtext
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# --- Physical Constants (SI Units) ---
k_B = 1.380649e-23     # J/K
T = 1e6                # Gas temperature (K)
G = 6.67430e-11        # m^3 / (kg s^2)
m_p = 1.67262192e-27   # Proton mass (kg)
P_power = 60      # Field power (W)
c = 299792458.0        # Speed of light (m/s)
mu = 1.0 # for normal hydrogen helium mix gas - in the 20K temp regime we use

# --- Conversion Factors ---
pc_to_m = 3.086e16
kpc_to_m = 3.086e19
kg_to_Msun = 1.0 / 1.98847e30
kg_to_amu = 1.0 / 1.66053906660e-27  # Convert kg to atomic mass units
kg_m3_to_amu_cm3 = kg_to_amu / 1e6   # Convert kg/m^3 to amu/cm^3

# --- Boundary Conditions ---
N0_cm3 = 0.5  
N0 = N0_cm3 * 1e6  # particles/m^3
C_dark = P_power / (c**3)

# --- Density Thresholds ---
min_density_cm3 = 0.0
max_density_cm3 = 0.01 # at densities over this, power is P_power
 
N_min = min_density_cm3 * 1e6  # Convert to particles/m^3
N_max = max_density_cm3 * 1e6  
# def dm_mass(N): 
#     if N <= N_min:
#         print(f"rho = 0 , N={N}, fraction={0}")
#         return 0
#     if N >= N_max:
#         print(f"rho = {C_dark * (N**(2/3))} , N={N}, fraction={1}")
#         return C_dark * (N**(2/3))

#     #interpolate between these points
#     range = N_max - N_min
#     fraction = (N - N_min) / range
#     rho = C_dark * np.sqrt(fraction) * (N**(2/3))
#     print(f"rho = {rho} , N={N}, fraction={fraction}")
#     return rho

# def d_dm_mass_dN(N):
#     if N <= N_min:
#         return 0
#     if N >= N_max:
#         return (2/3) * C_dark * (N**(-1/3))

#      # $$f'(N) = \frac{\text{CONST} \cdot (7N - 4N_m)}{6 \cdot \sqrt{R_c} \cdot \sqrt{N - N_m} \cdot N^{1/3}}$$
#     # did the derivate with LLM
#     range = N_max - N_min
#     top = C_dark*(7*N - 4*N_min) 
#     bottom = 6 * np.sqrt(range)*np.sqrt(N - N_min)*N**(1/3)
#     return top/bottom


# LINEAR
def dm_mass(N): 
    if N <= N_min:
        #print(f"rho = 0 , N={N}, fraction={0}")
        return 0
    if N >= N_max:
        #print(f"rho = {C_dark * (N**(2/3))} , N={N}, fraction={1}")
        return C_dark * (N**(2/3))

    #interpolate between these points
    range = N_max - N_min
    fraction = (N - N_min) / range
    rho = C_dark * fraction * (N**(2/3))
    #print(f"rho = {rho} , N={N}, fraction={fraction}")
    return rho

def d_dm_mass_dN(N):
    if N <= N_min:
        return 0
    if N >= N_max:
        return (2/3) * C_dark * (N**(-1/3))

     # $$f'(N) = \frac{\text{CONST} \cdot (7N - 4N_m)}{6 \cdot \sqrt{R_c} \cdot \sqrt{N - N_m} \cdot N^{1/3}}$$
    # did the derivate with LLM
    range = N_max - N_min
    top = C_dark*(5*N - 2*N_min) 
    bottom = 3 * range * N**(1/3)
    return top/bottom



# --- Density Functions (with Cutoff) ---
def rho(N):
    """Total mass density (kg/m^3)."""
    return m_p * mu * N + dm_mass(N)

def drho_dN(N):
    """Derivative of total mass density w.r.t N."""
    return m_p * mu + d_dm_mass_dN(N)

# Vectorized version of rho for processing the final arrays
def rho_array(N_arr):
    res = np.zeros_like(N_arr)
    mask = N_arr < N_min
    res[mask] = m_p * mu * N_arr[mask]
    res[~mask] = m_p * mu * N_arr[~mask] + C_dark * (N_arr[~mask]**(2/3))
    return res

# --- ODE System ---
def hydrostatic_ode(r, y):
    N, dN_dr, M_gas, M_tot = y
    
    # if N <= N_min:
    #     N = N_min
    #     dN_dr = 0
 
    # if N <= 1e-14:
    #     return [0, 0, 0, 0]
        
    term1 = - (2.0 / r) * dN_dr
    term2 = (drho_dN(N) / rho(N)) * (dN_dr**2)
    term3 = - (4 * np.pi * G / (k_B * T)) * (rho(N)**2)
    
    d2N_dr2 = term1 + term2 + term3
    
    # dM/dr for Enclosed Mass
    dM_gas_dr = 4 * np.pi * (r**2) * m_p * mu * N
    dM_tot_dr = 4 * np.pi * (r**2) * rho(N)
    

    retarray = [dN_dr, d2N_dr2, dM_gas_dr, dM_tot_dr]
    # if any of the retarray is NAN or Inf, then print it and the arguments 
    for i, val in enumerate(retarray):
        if np.isnan(val) or np.isinf(val):
            print(f"NAN or Inf detected at r={r}, y={y}, index={i}")

    return retarray


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

# --- Execute Solver --- 'Radau', 'RK45'
solution = solve_ivp(hydrostatic_ode, r_span, y0, method='RK45', events=cloud_edge, rtol=1e-10)

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
ax1.set_xscale("log")
ax1.set_xlim(0.02, 10)
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

