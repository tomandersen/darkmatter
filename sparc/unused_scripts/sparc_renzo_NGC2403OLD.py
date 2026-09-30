import numpy as np
from scipy.integrate import quad, trapezoid
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

# 1. Load and parse the SPARC data for NGC 2403
filename = './sparc/MassModels_Lelli2016c.mrt'
data = []

# Parse the text file manually to ignore header data and extract only NGC 2403
with open(filename, 'r') as f:
    for line in f:
        if line.startswith("NGC2403"):
            parts = line.split()
            # Extract R, Vobs, e_Vobs, Vgas, Vdisk, Vbul based on SPARC columns
            # Column indices: R(2), Vobs(3), e_Vobs(4), Vgas(5), Vdisk(6), Vbul(7)
            data.append([float(parts[2]), float(parts[3]), float(parts[4]), 
                         float(parts[5]), float(parts[6]), float(parts[7])])

data = np.array(data)
R_kpc = data[:, 0]
V_obs_kms = data[:, 1]
e_Vobs_kms = data[:, 2]
V_gas_kms = data[:, 3]
V_disk_kms = data[:, 4]
V_bul_kms = data[:, 5]

# 2. Define Physics Constants and Geometry
G = 6.67430e-11        # Gravitational constant (m^3 / kg s^2)
c = 2.9979e8           # Speed of light (m/s)
m_p = 1.67262e-27      # Proton mass (kg)
pc_to_m = 3.086e16     # Parsec to meters
kpc_to_m = pc_to_m * 1000

hz_kpc = 0.3           # Assumed 300 pc scale height for thick disk
hz_m = hz_kpc * kpc_to_m

R_m = R_kpc * kpc_to_m
V_gas_ms = V_gas_kms * 1000.0
V_disk_ms = V_disk_kms * 1000.0

# 3. Derive 3D Baryonic Density
valid = R_m > 0  # Avoid division by zero
Sigma_gas = np.zeros_like(R_m)
Sigma_disk = np.zeros_like(R_m)

# Approximate surface mass density (kg/m^2) from velocity components
Sigma_gas[valid] = (V_gas_ms[valid]**2) / (2 * np.pi * G * R_m[valid])
Sigma_disk[valid] = (V_disk_ms[valid]**2) / (2 * np.pi * G * R_m[valid])
Sigma_total = Sigma_gas + Sigma_disk

# Volume mass density (kg/m^3) and Number density (baryons/m^3)
rho_b = Sigma_total / (2 * hz_m)
n_R = rho_b / m_p 

# 4. Integrate Thick Disk Dark Mass for Base Power (P=1 Watt)
P_base = 1.0
rho_dm_base = (P_base / c**3) * (n_R ** (2/3))  # kg/m^3
Sigma_dm_base = rho_dm_base * (2 * hz_m)        # kg/m^2

V_dm_sq_base = np.zeros_like(R_m)

for i, R_i in enumerate(R_m):
    if R_i == 0:
        continue
        
    radial_integrand = np.zeros_like(R_m)
    for j, R_j in enumerate(R_m):
        if R_j == 0:
            continue
            
        def theta_integrand(theta):
            numerator = R_i - R_j * np.cos(theta)
            denominator = (R_i**2 + R_j**2 - 2 * R_i * R_j * np.cos(theta) + hz_m**2)**1.5
            return numerator / denominator
            
        ang_int, _ = quad(theta_integrand, 0, 2 * np.pi)
        radial_integrand[j] = Sigma_dm_base[j] * R_j * ang_int
        
    V_dm_sq_val = G * R_i * trapezoid(radial_integrand, R_m)
    V_dm_sq_base[i] = max(0, V_dm_sq_val)

V_dm_sq_base_kms = V_dm_sq_base / (1000.0**2)

# 5. Optimize the Power Parameter P
def calculate_chi2(P_test):
    # Scale base integral by P_test
    V_dm_test_sq = P_test * V_dm_sq_base_kms
    
    # Combine baryonic and dark mass velocities in quadrature
    V_tot_test = np.sqrt(V_disk_kms**2 + V_gas_kms**2 + V_bul_kms**2 + V_dm_test_sq)
    
    # Calculate Chi-squared against observed SPARC data
    chi2 = np.sum(((V_obs_kms[valid] - V_tot_test[valid]) / e_Vobs_kms[valid])**2)
    return chi2

result = minimize_scalar(calculate_chi2, bounds=(2.0, 10.0), method='bounded')
best_P = result.x
min_chi2 = result.fun

print(f"Optimization Complete!")
print(f"Best fit Power (P): {best_P:.4f} Watts")
print(f"Minimum Chi-squared: {min_chi2:.2f}")

# 6. Plot the Final Optimized Model
V_dm_opt_kms = np.sqrt(best_P * V_dm_sq_base_kms)
V_total_opt_kms = np.sqrt(V_disk_kms**2 + V_gas_kms**2 + V_bul_kms**2 + V_dm_opt_kms**2)

plt.figure(figsize=(10, 6))
plt.plot(R_kpc, V_gas_kms, 'b:', label='Gas (SPARC)')
plt.plot(R_kpc, V_disk_kms, 'y:', label='Stars (SPARC)')
plt.plot(R_kpc, V_dm_opt_kms, 'm--', label=f'Dark Mass (Optimized P={best_P:.2f} W)')
plt.plot(R_kpc, V_total_opt_kms, 'r-', linewidth=2, label='Total Predicted')
plt.errorbar(R_kpc, V_obs_kms, yerr=e_Vobs_kms, fmt='ko', label='Observed Data', capsize=3)

plt.xlabel('Radius (kpc)')
plt.ylabel('Velocity (km/s)')
plt.title(f'NGC 2403: Thick Disk Dark Mass Theory (Best Fit P = {best_P:.2f} W)')
plt.legend()
plt.grid(True)
plt.show()