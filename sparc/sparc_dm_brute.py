import numpy as np
import os
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import warnings

# Suppress integration warnings for highly dense inner rings
warnings.filterwarnings("ignore")

Upsilon_disk = 0.5   
Upsilon_bulge = 0.7  

sqrt_Upsilon_disk = np.sqrt(Upsilon_disk)
sqrt_Upsilon_bulge = np.sqrt(Upsilon_bulge)

MIN_DENSITY_FOR_DM = 0
MIN_DENSITY_WITHIN_R_D = 0   


NUM_BLOBS_THETA = 360
NUM_LAYERS_Z = 25
USE_EXPONENTIAL_DISK = True

def get_sparc_galaxy_scale_height(radius_kpc, R_d):
    """Calculates the vertical disk scale height (thickness) of a SPARC galaxy."""
    z_d = 0.196 * (R_d ** 0.633)
    return z_d

# 1. Setup and Parse the Data
properties_file = "./sparc/SPARC_Lelli2016c.mrt.txt"
kinematics_file = "./sparc/MassModels_Lelli2016_SigmaGas.mrt"
output_galaxies = "./sparc/dm_mass/galaxies"
output_dir = "./sparc/dm_mass"
os.makedirs(output_galaxies, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

rd_dict = {}
with open(properties_file, 'r') as f:
    for line in f:
        parts = line.split()
        if len(parts) >= 12 and parts[1].lstrip('-').isdigit():
            gal_name = parts[0]
            try:
                r_d = float(parts[11])
                rd_dict[gal_name] = r_d
            except ValueError:
                continue

galaxies = {}

with open(kinematics_file, 'r') as f:
    for line in f:
        parts = line.split()
        if len(parts) == 11 and not line.startswith('---') and not line.startswith('Byte'):
            try:
                gal_name = parts[0]
                R = float(parts[2])
                Vobs = float(parts[3])
                eVobs = float(parts[4])
                Vgas = float(parts[5])
                Vdisk = float(parts[6])
                Vbul = float(parts[7])
                Sigma_gas = float(parts[10]) # HI particle surface density per cm^2
                
                Vdisk = Vdisk * sqrt_Upsilon_disk
                Vbul = Vbul * sqrt_Upsilon_bulge

                if eVobs <= 0:
                    eVobs = 1.0 
                    
                if gal_name in rd_dict:
                    if gal_name not in galaxies:
                        galaxies[gal_name] = {'R': [], 'Vobs': [], 'eVobs': [], 'Vgas': [], 'Vdisk': [], 'Vbul': [], 'Sigma_gas': []}
                    
                    galaxies[gal_name]['R'].append(R)
                    galaxies[gal_name]['Vobs'].append(Vobs)
                    galaxies[gal_name]['eVobs'].append(eVobs)
                    galaxies[gal_name]['Vgas'].append(Vgas)
                    galaxies[gal_name]['Vdisk'].append(Vdisk)
                    galaxies[gal_name]['Vbul'].append(Vbul)
                    galaxies[gal_name]['Sigma_gas'].append(Sigma_gas)
                    
            except ValueError:
                continue

for gal in galaxies:
    for key in galaxies[gal]:
        galaxies[gal][key] = np.array(galaxies[gal][key])

# 2. Physics Constants
G = 6.67430e-11        
c = 2.9979e8           
m_p = 1.67262e-27      
pc_to_m = 3.086e16     
kpc_to_m = pc_to_m * 1000
a0_mond = 1.2e-10      

# 3. Pre-calculate Base Integrals (P=1 Watt) for ALL galaxies
print(f"Pre-calculating base arrays for {len(galaxies)} galaxies via 3D brute force...")

for i, gal in enumerate(galaxies):
    print(f"Processing {gal} ({i+1}/{len(galaxies)})...", end='\r')
    
    g_data = galaxies[gal]
    R_m = g_data['R'] * kpc_to_m
    V_gas_ms = g_data['Vgas'] * 1000.0
    V_disk_ms = g_data['Vdisk'] * 1000.0
    V_bul_ms = g_data['Vbul'] * 1000.0
    Sigma_gas_cm2 = g_data['Sigma_gas'] 
    
    R_d = rd_dict[gal]
    hz_kpc = get_sparc_galaxy_scale_height(None, R_d)
    hz_m = hz_kpc * kpc_to_m
    
    valid = R_m > 0
    Sigma_gas_m2 = Sigma_gas_cm2 * 100.0 * 100.0
    n_R_gas = Sigma_gas_m2 / (2 * hz_m) 
    n_R_gas = n_R_gas * 1.0833 # multiplies by 1.0833 to include He and other particle counts... 
    
    min_n_global = MIN_DENSITY_FOR_DM * 1e6
    min_n_inner = MIN_DENSITY_WITHIN_R_D * 1e6
    
    valid_dm_density = np.ones_like(n_R_gas, dtype=bool)
    
    if MIN_DENSITY_FOR_DM > 0:
        valid_dm_density &= (n_R_gas >= min_n_global)
        
    if MIN_DENSITY_WITHIN_R_D > 0:
        inner_mask = g_data['R'] <= R_d
        valid_dm_density[inner_mask] &= (n_R_gas[inner_mask] >= min_n_inner)

    rho_dm_base = np.zeros_like(n_R_gas)
    rho_dm_base[valid_dm_density] = (1.0 / c**3) * (n_R_gas[valid_dm_density] ** (2/3))
    
    # ---------------------------------------------------------
    # BRUTE FORCE 3D BLOB CALCULATION
    # ---------------------------------------------------------
    nR = len(R_m)
    nz = NUM_LAYERS_Z
    ntheta = NUM_BLOBS_THETA
    
    # R grid dimensions
    dR = np.gradient(R_m)
    
    # Z grid: NUM_LAYERS_Z layers from -2.0*hz_m to +2.0*hz_m
    Z_m = np.linspace(-2.0 * hz_m, 2.0 * hz_m, nz)
    dZ = Z_m[1] - Z_m[0] if nz > 1 else 2.0 * hz_m
    
    # Theta grid: ntheta blobs around the z-axis (0 to 360 degrees)
    theta_rad = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
    dTheta = 2 * np.pi / ntheta
    
    # Create 3D meshgrids (Vectorized equivalent of nested loops)
    R_grid, Z_grid, Theta_grid = np.meshgrid(R_m, Z_m, theta_rad, indexing='ij')
    dR_grid = dR[:, None, None]
    
    # Blob volume
    dV = R_grid * dTheta * dR_grid * dZ
    
    # Map 1D density constraints into the 3D grid 
    gas_particle_count_He_factor = (1.33)**3 # This SHOULD be 1.33....
    Sigma_gas_kg_m2_for_Vgas = Sigma_gas_m2 * m_p*gas_particle_count_He_factor # (should be 1.33 I think) accounts for density of the Helium, etc
    gas_density_1d = np.where(R_m > 0, Sigma_gas_kg_m2_for_Vgas / (2.0 * hz_m), 0.0)
    dm_density_1d = rho_dm_base 
    
    gas_density_grid = gas_density_1d[:, None, None] * np.ones_like(Z_grid)
    dm_density_grid = dm_density_1d[:, None, None] * np.ones_like(Z_grid)

    # ----------------------------------------------------------------------
    # Apply density profile (Hard Cut or Exponential Falloff)
    # ----------------------------------------------------------------------
    if USE_EXPONENTIAL_DISK:
        # Exponential falloff in Z: rho ~ exp(-2/3 |z| / h_z)
        gas_density_grid *= 0.5*np.exp(-(2.0/3.0) * np.abs(Z_grid) / hz_m)
    else:
        # Hard cut at z = +/- h_z
        z_mask = np.abs(Z_grid) <= hz_m
        gas_density_grid[~z_mask] = 0.0
        dm_density_grid[~z_mask] = 0.0

    
    # Blob Masses
    dm_gas = gas_density_grid * dV
    dm_dm = dm_density_grid * dV
    
    # Cartesian positions of all blobs
    X_blob = R_grid * np.cos(Theta_grid)
    Y_blob = R_grid * np.sin(Theta_grid)
    Z_blob = Z_grid
    
    V_gas_sq = np.zeros_like(R_m)
    V_dm_sq_base = np.zeros_like(R_m)
    
    # Softening prevents infinite acceleration if a test point shares exact coordinates with a blob center
    softening_sq = (0.01 * hz_m)**2 
    
    # Loop over each target test point radius
    for i_idx, R_test in enumerate(R_m):
        if R_test == 0:
            continue
            
        # Distances from target test point (R_test, 0, 0) to all surrounding blobs
        dx = X_blob - R_test
        dy = Y_blob
        dz = Z_blob
        
        dist_sq = dx**2 + dy**2 + dz**2 + softening_sq
        dist = np.sqrt(dist_sq)
        
        # Calculate x-component of gravitational acceleration from each blob
        # da_x = G * dm * dx / dist^3
        da_x_factor = G * dx / (dist_sq * dist)
        
        a_x_gas = np.sum(da_x_factor * dm_gas)
        a_x_dm = np.sum(da_x_factor * dm_dm)
        
        # Inward radial acceleration is -a_x. Since V^2 / R = a_inward, then V^2 = -a_x * R_test
        V_gas_sq_val = -a_x_gas * R_test
        V_dm_sq_val = -a_x_dm * R_test
        
        V_gas_sq[i_idx] = V_gas_sq_val
        V_dm_sq_base[i_idx] = max(0, V_dm_sq_val)
        
    g_data['V_dm_sq_base_kms'] = V_dm_sq_base / (1000.0**2)
    g_data['Vgas_test'] = np.sign(V_gas_sq) * np.sqrt(np.abs(V_gas_sq)) / 1000.0
    # ---------------------------------------------------------
     
    V_bar_sq = np.clip(V_disk_ms**2 + V_bul_ms**2 + V_gas_ms * np.abs(V_gas_ms), 0, None)
    g_N = np.zeros_like(R_m)
    g_N[valid] = V_bar_sq[valid] / R_m[valid]
    
    g_mond = (g_N + np.sqrt(g_N**2 + 4 * g_N * a0_mond)) / 2.0
    g_data['V_mond_kms'] = np.sqrt(g_mond * R_m) / 1000.0

print("\nIntegrations complete. Optimizing global parameter for Linear Absolute Error...")

# 4. Global Optimization Function
def global_linear_err(P_test):
    total_linear = 0
    for gal in galaxies:
        g = galaxies[gal]
        V_dm_test_sq = P_test * g['V_dm_sq_base_kms']
        
        V_bar_sq_kms = np.clip(g['Vdisk']**2 + g['Vbul']**2 + g['Vgas'] * np.abs(g['Vgas']), 0, None)
        V_tot_test = np.sqrt(V_bar_sq_kms + V_dm_test_sq)
        
        linear_err = np.sum(np.abs(g['Vobs'] - V_tot_test) / g['eVobs'])
        total_linear += linear_err
        
    return total_linear

result = minimize_scalar(global_linear_err, bounds=(2.0, 100.0), method='bounded')
best_global_P = result.x
min_linear_err = result.fun

dm_chi2_total = 0
mond_linear_total = 0
mond_chi2_total = 0

for gal in galaxies:
    g = galaxies[gal]
    V_dm_final_kms = np.sqrt(best_global_P * g['V_dm_sq_base_kms'])
    V_bar_sq_kms = np.clip(g['Vdisk']**2 + g['Vbul']**2 + g['Vgas'] * np.abs(g['Vgas']), 0, None)
    V_tot_final_kms = np.sqrt(V_bar_sq_kms + V_dm_final_kms**2)
    
    dm_chi2_total += np.sum(((g['Vobs'] - V_tot_final_kms) / g['eVobs'])**2)
    
    mond_linear_total += np.sum(np.abs(g['Vobs'] - g['V_mond_kms']) / g['eVobs'])
    mond_chi2_total += np.sum(((g['Vobs'] - g['V_mond_kms']) / g['eVobs'])**2)

print(f"\n--- GLOBAL OPTIMIZATION RESULTS ---")
print(f"Best Universal Power (P): {best_global_P:.4f} Watts")
print(f"Dark Mass Theory - Minimum Linear Error:  {min_linear_err:.2f}")
print(f"Dark Mass Theory - Resulting Chi-squared: {dm_chi2_total:.2f}")
print(f"\n--- MOND (Simple) PREDICTION ERRORS ---")
print(f"MOND - Linear Error:  {mond_linear_total:.2f}")
print(f"MOND - Chi-squared:   {mond_chi2_total:.2f}")

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
    plt.plot(g['R'], g['Vgas_test'], 'k:', label='Gas_test')
    if np.any(g['Vbul'] > 0):
        plt.plot(g['R'], g['Vbul'], 'g:', label='Bulge')
    plt.plot(g['R'], V_dm_final_kms, 'm--', label=f'Dark Mass (P={best_global_P:.2f}W)')
    
    plt.plot(g['R'], g['V_mond_kms'], 'c-.', linewidth=2, label='MOND (Simple)')
    plt.plot(g['R'], V_tot_final_kms, 'r-', linewidth=2, label='Total DM Predicted')
    plt.errorbar(g['R'], g['Vobs'], yerr=g['eVobs'], fmt='ko', label='Observed Data', capsize=2)
    
    plt.xlabel('Radius (kpc)')
    plt.ylabel('Velocity (km/s)')
    plt.title(f'{gal} Kinematics (Universal P = {best_global_P:.2f} W)')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(os.path.join(output_galaxies, f"{gal}_dm_mass.png"), bbox_inches='tight', dpi=300)
    plt.close()

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

plt.savefig(os.path.join(output_dir, "global_fit_vs_obs.png"), bbox_inches='tight', dpi=300)
plt.close()

print("\nGenerating global acceleration scatter plot...")

all_g_obs = []
all_g_pred = []

for gal in galaxies:
    g = galaxies[gal]
    R_m = g['R'] * kpc_to_m
    valid = R_m > 0 
    
    V_dm_final_sq = best_global_P * g['V_dm_sq_base_kms']
    V_bar_sq_kms = np.clip(g['Vdisk']**2 + g['Vbul']**2 + np.sign(g['Vgas'])*(g['Vgas']**2), 0, None)
    
    V_tot_final_sq_ms = (V_bar_sq_kms + V_dm_final_sq) * (1000.0**2)
    V_obs_sq_ms = (g['Vobs'] * 1000.0)**2
    
    g_obs = V_obs_sq_ms[valid] / R_m[valid]
    g_pred = V_tot_final_sq_ms[valid] / R_m[valid]
    
    all_g_obs.extend(g_obs)
    all_g_pred.extend(g_pred)

plt.figure(figsize=(9, 9))
plt.scatter(all_g_obs, all_g_pred, alpha=0.3, edgecolors='none', c='blue')

plt.xscale('log')
plt.yscale('log')

min_val = 10**np.floor(np.log10(min(min(all_g_obs), min(all_g_pred))))
max_val = 10**np.ceil(np.log10(max(max(all_g_obs), max(all_g_pred))))

plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Newton')

plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)
plt.axis('square')

plt.xlabel('Observed Acceleration $g_{obs}$ (m/s$^2$)', fontsize=12)
plt.ylabel('Predicted Acceleration $g_{pred}$ (m/s$^2$)', fontsize=12)
plt.title(f'Global Fit: Dark Mass vs Observed Acceleration\n(Total Points = {len(all_g_obs)}, Universal P = {best_global_P:.2f} W)', fontsize=14)
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)

plt.savefig(os.path.join(output_dir, "global_predicted_accel_vs_obs.png"), bbox_inches='tight', dpi=200)
plt.close()

print(f"All operations completed successfully. Files saved to '/{output_dir}'.")