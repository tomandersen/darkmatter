import numpy as np
import os
from scipy.integrate import quad, trapezoid
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import warnings

# Suppress integration warnings for highly dense inner rings
warnings.filterwarnings("ignore")

# 1. Setup and Parse the Data
filename = './sparc/MassModels_Lelli2016c.mrt'
output_dir = "renzo"
os.makedirs(output_dir, exist_ok=True)

galaxies = {}

# Parse the dataset, separating it by galaxy
with open(filename, 'r') as f:
    for line in f:
        parts = line.split()
        # Look for lines with exactly 10 columns where Radius (col 3) is a number
        if len(parts) == 10:
            try:
                gal_name = parts[0]
                R = float(parts[2])
                Vobs = float(parts[3])
                eVobs = float(parts[4])
                Vgas = float(parts[5])
                Vdisk = float(parts[6])
                Vbul = float(parts[7])
                
                # Prevent division by zero if error is listed as 0.0
                if eVobs <= 0:
                    eVobs = 1.0 
                    
                if gal_name not in galaxies:
                    galaxies[gal_name] = {'R': [], 'Vobs': [], 'eVobs': [], 'Vgas': [], 'Vdisk': [], 'Vbul': []}
                
                galaxies[gal_name]['R'].append(R)
                galaxies[gal_name]['Vobs'].append(Vobs)
                galaxies[gal_name]['eVobs'].append(eVobs)
                galaxies[gal_name]['Vgas'].append(Vgas)
                galaxies[gal_name]['Vdisk'].append(Vdisk)
                galaxies[gal_name]['Vbul'].append(Vbul)
            except ValueError:
                # Skip header lines that cannot be converted to floats
                continue

# Convert lists to numpy arrays
for gal in galaxies:
    for key in galaxies[gal]:
        galaxies[gal][key] = np.array(galaxies[gal][key])

# 2. Physics Constants
G = 6.67430e-11        # Gravitational constant (m^3 / kg s^2)
c = 2.9979e8           # Speed of light (m/s)
m_p = 1.67262e-27      # Proton mass (kg)
pc_to_m = 3.086e16     # Parsec to meters
kpc_to_m = pc_to_m * 1000

# Using a standard 300 pc universal thick-disk proxy for the entire sweep
hz_m = 0.3 * kpc_to_m 

# 3. Pre-calculate Base Integrals (P=1 Watt) for ALL galaxies
print(f"Pre-calculating base integrals for {len(galaxies)} galaxies. This may take a moment...")

for i, gal in enumerate(galaxies):
    print(f"Processing {gal} ({i+1}/{len(galaxies)})...", end='\r')
    
    R_m = galaxies[gal]['R'] * kpc_to_m
    V_gas_ms = galaxies[gal]['Vgas'] * 1000.0
    V_disk_ms = galaxies[gal]['Vdisk'] * 1000.0
    V_bul_ms = galaxies[gal]['Vbul'] * 1000.0
    
    # Isolate total baryonic mass to find n(R)
    valid = R_m > 0
    Sigma_b = np.zeros_like(R_m)
    Sigma_b[valid] = (V_gas_ms[valid]**2 + V_disk_ms[valid]**2 + V_bul_ms[valid]**2) / (2 * np.pi * G * R_m[valid])
    
    rho_b = Sigma_b / (2 * hz_m)
    n_R = rho_b / m_p 
    
    # Calculate Base Dark Mass Profile (P=1)
    rho_dm_base = (1.0 / c**3) * (n_R ** (2/3))
    Sigma_dm_base = rho_dm_base * (2 * hz_m)
    
    V_dm_sq_base = np.zeros_like(R_m)
    
    for i_idx, R_i in enumerate(R_m):
        if R_i == 0:
            continue
            
        radial_integrand = np.zeros_like(R_m)
        for j_idx, R_j in enumerate(R_m):
            if R_j == 0:
                continue
                
            def theta_integrand(theta):
                numerator = R_i - R_j * np.cos(theta)
                denominator = (R_i**2 + R_j**2 - 2 * R_i * R_j * np.cos(theta) + hz_m**2)**1.5
                return numerator / denominator
                
            ang_int, _ = quad(theta_integrand, 0, 2 * np.pi)
            radial_integrand[j_idx] = Sigma_dm_base[j_idx] * R_j * ang_int
            
        V_dm_sq_val = G * R_i * trapezoid(radial_integrand, R_m)
        V_dm_sq_base[i_idx] = max(0, V_dm_sq_val)
        
    # Store base squared velocity in km/s
    galaxies[gal]['V_dm_sq_base_kms'] = V_dm_sq_base / (1000.0**2)

print("\nIntegrations complete. Optimizing global parameter...")

# 4. Global Optimization Function
def global_chi2(P_test):
    total_chi2 = 0
    for gal in galaxies:
        g = galaxies[gal]
        
        # Scale by P
        V_dm_test_sq = P_test * g['V_dm_sq_base_kms']
        
        # Total predicted velocity
        V_tot_test = np.sqrt(g['Vdisk']**2 + g['Vgas']**2 + g['Vbul']**2 + V_dm_test_sq)
        
        # Add to global Chi-squared sum
        chi2 = np.sum(((g['Vobs'] - V_tot_test) / g['eVobs'])**2)
        total_chi2 += chi2
        
    return total_chi2

# Find the best universal Power
result = minimize_scalar(global_chi2, bounds=(2.0, 10.0), method='bounded')
best_global_P = result.x

print(f"\n--- GLOBAL OPTIMIZATION RESULTS ---")
print(f"Best Universal Power (P): {best_global_P:.4f} Watts")
print(f"Global Minimum Chi-squared: {result.fun:.2f}")

# 5. Generate Output Plots
print("\nGenerating 175 individual plots and master summary...")
all_V_obs = []
all_V_pred = []

for gal in galaxies:
    g = galaxies[gal]
    
    # Calculate finalized velocities using the best global P
    V_dm_final_kms = np.sqrt(best_global_P * g['V_dm_sq_base_kms'])
    V_tot_final_kms = np.sqrt(g['Vdisk']**2 + g['Vgas']**2 + g['Vbul']**2 + V_dm_final_kms**2)
    
    # Store for global scatter plot
    all_V_obs.extend(g['Vobs'])
    all_V_pred.extend(V_tot_final_kms)
    
    # Generate Individual Plot
    plt.figure(figsize=(8, 5))
    plt.plot(g['R'], g['Vgas'], 'b:', label='Gas')
    plt.plot(g['R'], g['Vdisk'], 'y:', label='Stars')
    if np.any(g['Vbul'] > 0):
        plt.plot(g['R'], g['Vbul'], 'g:', label='Bulge')
    plt.plot(g['R'], V_dm_final_kms, 'm--', label=f'Dark Mass (P={best_global_P:.2f}W)')
    
    plt.plot(g['R'], V_tot_final_kms, 'r-', linewidth=2, label='Total Predicted')
    plt.errorbar(g['R'], g['Vobs'], yerr=g['eVobs'], fmt='ko', label='Observed Data', capsize=3)
    
    plt.xlabel('Radius (kpc)')
    plt.ylabel('Velocity (km/s)')
    plt.title(f'{gal} Kinematics (Universal P = {best_global_P:.2f} W)')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(os.path.join(output_dir, f"{gal}_kinematics.png"), bbox_inches='tight')
    plt.close()

# Generate Master Scatter Plot
plt.figure(figsize=(9, 9))
plt.scatter(all_V_obs, all_V_pred, alpha=0.3, edgecolors='none', c='blue')

# 1-to-1 reference line
max_val = max(max(all_V_obs), max(all_V_pred))
plt.plot([0, max_val], [0, max_val], 'k--', linewidth=2, label='Perfect Fit (1:1)')

plt.xlabel('Observed Velocity (km/s)', fontsize=12)
plt.ylabel('Predicted Velocity (km/s)', fontsize=12)
plt.title(f'Global Fit: Dark Mass Theory vs SPARC Dataset\n(Total Points = {len(all_V_obs)}, Universal P = {best_global_P:.2f} W)', fontsize=14)
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.xlim(0, max_val + 20)
plt.ylim(0, max_val + 20)

plt.savefig(os.path.join(output_dir, "global_fit_vs_obs.png"), bbox_inches='tight')
plt.close()

print(f"All operations completed successfully. Files saved to '/{output_dir}'.")