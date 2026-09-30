import numpy as np
import os
from scipy.integrate import quad, trapezoid
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import warnings

# Suppress integration warnings for highly dense inner rings
warnings.filterwarnings("ignore")

def get_sparc_galaxy_scale_height(radius_kpc, R_d):
    """
    Calculates the vertical disk scale height (thickness) of a SPARC galaxy.
    """
    # Empirically derived scale-height relation used for SPARC mass modeling
    z_d = 0.196 * (R_d ** 0.633)
    return z_d

# 1. Setup and Parse the Data
properties_file = "./sparc/SPARC_Lelli2016c.mrt.txt"
kinematics_file = "./sparc/MassModels_Lelli2016c.mrt"
output_dir = "renzo_rd"
os.makedirs(output_dir, exist_ok=True)

# Parse the properties file to extract R_d (disk scale length) for each galaxy
rd_dict = {}
with open(properties_file, 'r') as f:
    for line in f:
        parts = line.split()
        # Heuristic to find data rows: check if second column is an integer (Hubble type)
        if len(parts) >= 12 and parts[1].lstrip('-').isdigit():
            gal_name = parts[0]
            try:
                r_d = float(parts[11])
                rd_dict[gal_name] = r_d
            except ValueError:
                continue

galaxies = {}

# Parse the kinematics file
with open(kinematics_file, 'r') as f:
    for line in f:
        parts = line.split()
        if len(parts) == 10 and not line.startswith('---') and not line.startswith('Byte'):
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
                    
                # Only include galaxies that have a known R_d from the properties file
                if gal_name in rd_dict:
                    if gal_name not in galaxies:
                        galaxies[gal_name] = {'R': [], 'Vobs': [], 'eVobs': [], 'Vgas': [], 'Vdisk': [], 'Vbul': []}
                    
                    galaxies[gal_name]['R'].append(R)
                    galaxies[gal_name]['Vobs'].append(Vobs)
                    galaxies[gal_name]['eVobs'].append(eVobs)
                    galaxies[gal_name]['Vgas'].append(Vgas)
                    galaxies[gal_name]['Vdisk'].append(Vdisk)
                    galaxies[gal_name]['Vbul'].append(Vbul)
            except ValueError:
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
a0_mond = 1.2e-10      # Standard MOND acceleration constant (m/s^2)

# 3. Pre-calculate Base Integrals (P=1 Watt) for ALL galaxies
print(f"Pre-calculating base integrals for {len(galaxies)} galaxies...")

for i, gal in enumerate(galaxies):
    print(f"Processing {gal} ({i+1}/{len(galaxies)})...", end='\r')
    
    g_data = galaxies[gal]
    R_m = g_data['R'] * kpc_to_m
    V_gas_ms = g_data['Vgas'] * 1000.0
    V_disk_ms = g_data['Vdisk'] * 1000.0
    V_bul_ms = g_data['Vbul'] * 1000.0
    
    # Calculate galaxy-specific scale height
    R_d = rd_dict[gal]
    hz_kpc = get_sparc_galaxy_scale_height(None, R_d)
    hz_m = hz_kpc * kpc_to_m
    
    valid = R_m > 0
    Sigma_b = np.zeros_like(R_m)
    
    # Estimate baryonic surface mass density (using sign preservation for gas holes)
    V_bar_sq = np.clip(V_disk_ms**2 + V_bul_ms**2 + V_gas_ms * np.abs(V_gas_ms), 0, None)
    Sigma_b[valid] = V_bar_sq[valid] / (2 * np.pi * G * R_m[valid])
    
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
        
    g_data['V_dm_sq_base_kms'] = V_dm_sq_base / (1000.0**2)
    
    # MOND Prediction Calculation
    g_N = np.zeros_like(R_m)
    g_N[valid] = V_bar_sq[valid] / R_m[valid]
    
    # Simple MOND interpolation function: mu(x) = x / (1+x) leads to the standard quadratic
    g_mond = (g_N + np.sqrt(g_N**2 + 4 * g_N * a0_mond)) / 2.0
    g_data['V_mond_kms'] = np.sqrt(g_mond * R_m) / 1000.0

print("\nIntegrations complete. Optimizing global parameter...")

# 4. Global Optimization Function
def global_chi2(P_test):
    total_chi2 = 0
    for gal in galaxies:
        g = galaxies[gal]
        V_dm_test_sq = P_test * g['V_dm_sq_base_kms']
        
        # Total predicted velocity (re-evaluating baryonic V^2 to handle gas holes properly)
        V_bar_sq_kms = np.clip(g['Vdisk']**2 + g['Vbul']**2 + g['Vgas'] * np.abs(g['Vgas']), 0, None)
        V_tot_test = np.sqrt(V_bar_sq_kms + V_dm_test_sq)
        
        chi2 = np.sum(((g['Vobs'] - V_tot_test) / g['eVobs'])**2)
        total_chi2 += chi2
        
    return total_chi2

result = minimize_scalar(global_chi2, bounds=(2.0, 10.0), method='bounded')
best_global_P = result.x

print(f"\n--- GLOBAL OPTIMIZATION RESULTS ---")
print(f"Best Universal Power (P): {best_global_P:.4f} Watts")
print(f"Global Minimum Chi-squared: {result.fun:.2f}")

# 5. Generate Output Plots
print("\nGenerating individual galaxy plots and master summary...")
all_V_obs = []
all_V_pred = []

for gal in galaxies:
    g = galaxies[gal]
    
    V_dm_final_kms = np.sqrt(best_global_P * g['V_dm_sq_base_kms'])
    V_bar_sq_kms = np.clip(g['Vdisk']**2 + g['Vbul']**2 + g['Vgas'] * np.abs(g['Vgas']), 0, None)
    V_tot_final_kms = np.sqrt(V_bar_sq_kms + V_dm_final_kms**2)
    
    all_V_obs.extend(g['Vobs'])
    all_V_pred.extend(V_tot_final_kms)
    
    plt.figure(figsize=(8, 5))
    plt.plot(g['R'], g['Vgas'], 'b:', label='Gas')
    plt.plot(g['R'], g['Vdisk'], 'y:', label='Stars')
    if np.any(g['Vbul'] > 0):
        plt.plot(g['R'], g['Vbul'], 'g:', label='Bulge')
    plt.plot(g['R'], V_dm_final_kms, 'm--', label=f'Dark Mass (P={best_global_P:.2f}W)')
    
    plt.plot(g['R'], g['V_mond_kms'], 'c-.', linewidth=2, label='MOND (Simple)')
    plt.plot(g['R'], V_tot_final_kms, 'r-', linewidth=2, label='Total DM Predicted')
    plt.errorbar(g['R'], g['Vobs'], yerr=g['eVobs'], fmt='ko', label='Observed Data', capsize=3)
    
    plt.xlabel('Radius (kpc)')
    plt.ylabel('Velocity (km/s)')
    plt.title(f'{gal} Kinematics (Universal P = {best_global_P:.2f} W)')
    plt.legend()
    plt.grid(True)
    
    # Save with doubled linear resolution (dpi=200)
    plt.savefig(os.path.join(output_dir, f"{gal}_kinematics.png"), bbox_inches='tight', dpi=200)
    plt.close()

# Generate Master Scatter Plot
plt.figure(figsize=(9, 9))
plt.scatter(all_V_obs, all_V_pred, alpha=0.3, edgecolors='none', c='blue')

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

plt.savefig(os.path.join(output_dir, "global_fit_vs_obs.png"), bbox_inches='tight', dpi=200)
plt.close()

print(f"All operations completed successfully. Files saved to '/{output_dir}'.")