import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- Physical Constants (SI Units) ---
G = 6.67430e-11          # Gravitational constant (m^3 kg^-1 s^-2)
k_B = 1.380649e-23       # Boltzmann constant (J/K)
m_H = 1.6735575e-27      # Mass of hydrogen atom (kg)
M_sun = 1.98847e30       # Solar mass (kg)
kpc = 3.085677581e19     # Kiloparsec in meters

# --- Input Parameters ---
T = 2e2               # Temperature in Kelvin
n_c = 0.5 * 1e6          # Central particle density (0.5 per cm^3 converted to m^-3)
mu = 1.0                 # Usually 0.6, but my other script uses 1, (and usually temps near 20000K - Mean molecular weight (approx for ionized primordial gas)

# Calculate central mass density (rho_c)
rho_c = n_c * mu * m_H

# Calculate the characteristic scale radius 'a'
# This dictates the physical size of the flat core
a = np.sqrt((k_B * T) / (4 * np.pi * G * mu * m_H * rho_c))

# --- Define the dimensionless ODE ---
# The Isothermal Lane-Emden equation: (1/xi^2) * d/dxi(xi^2 * dpsi/dxi) = exp(-psi)
# We break this 2nd-order ODE into a system of two 1st-order ODEs:
# y[0] = psi
# y[1] = dpsi/dxi
def isothermal_sphere(xi, y):
    psi = y[0]
    dpsi_dxi = y[1]
    
    # Avoid division by zero at exactly xi=0
    if xi == 0:
        return [0, 0]
    
    d2psi_dxi2 = np.exp(-psi) - (2.0 / xi) * dpsi_dxi
    return [dpsi_dxi, d2psi_dxi2]

# --- Boundary Conditions & Integration ---
# Start slightly off-center to avoid the 1/xi singularity, using Taylor expansion 
# for a flat core: psi(xi) ~ xi^2 / 6 and psi'(xi) ~ xi / 3
xi_0 = 1e-8
y0 = [ (xi_0**2) / 6.0, xi_0 / 3.0 ]

# Integrate out to xi = 50 (approx 50 core radii outward)
xi_span = (xi_0, 5.0)
xi_eval = np.linspace(xi_0, 5.0, 1000)

solution = solve_ivp(isothermal_sphere, xi_span, y0, t_eval=xi_eval, method='RK45', rtol=1e-8, atol=1e-8)

xi = solution.t
psi = solution.y[0]
dpsi_dxi = solution.y[1]

# --- Convert back to physical units ---
# Radius
radius_m = a * xi
radius_kpc = radius_m / kpc

# Density (rho = rho_c * exp(-psi))
# Converting back to particles per cm^3 for intuitive reading
density_kg_m3 = rho_c * np.exp(-psi)
density_cm3 = density_kg_m3 / (mu * m_H) / 1e6

# Enclosed Mass
# Using the identity: M(r) = 4 * pi * rho_c * a^3 * (xi^2 * dpsi/dxi)
mass_kg = 4 * np.pi * rho_c * (a**3) * (xi**2 * dpsi_dxi)
mass_msun = mass_kg / M_sun

# --- Plotting ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Density Profile
ax1.plot(radius_kpc, density_cm3, color='blue', linewidth=2)
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.set_xlim(1e-2, 10)

ax1.set_xlabel('Radius (kpc)', fontsize=12)
ax1.set_ylabel('Density (particles / cm$^3$)', fontsize=12)
ax1.set_title('Gas Density Profile', fontsize=14)
ax1.grid(True, which="both", ls="--", alpha=0.5)

# Plot 2: Enclosed Mass Profile
ax2.plot(radius_kpc, mass_msun, color='red', linewidth=2)
ax2.set_xlabel('Radius (kpc)', fontsize=12)
ax2.set_ylabel(r'Enclosed Mass ($M_\odot$)', fontsize=12)
ax2.set_title('Enclosed Mass vs Radius', fontsize=14)
ax2.grid(True, ls="--", alpha=0.7)
ax2.set_xscale('log')
ax2.set_xlim(1e-2, 10)
ax2.set_yscale('log')
ax2.set_ylim(1e4, 1e12)


plt.tight_layout()
plt.show()