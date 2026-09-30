import numpy as np
import os
from scipy.integrate import quad
from scipy.optimize import minimize_scalar, least_squares
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

def get_sparc_galaxy_scale_height(radius_kpc, R_d):
    """Calculates the vertical disk scale height (thickness) of a SPARC galaxy."""
    return 0.196 * (R_d ** 0.633)

# 1. Setup and Parse the Data
properties_file = "./sparc/SPARC_Lelli2016c.mrt.txt"
kinematics_file = "./sparc/MassModels_Lelli2016c.mrt"
output_dir = "./sparc/renzo_g" 
inverse_dir = "./sparc/renzo_missing_baryons"

os.makedirs(output_dir, exist_ok=True)
os.makedirs(inverse_dir, exist_ok=True)

# Parse the properties file to extract R_d
rd_dict = {}
with open(properties_file, 'r') as f:
    for line in f:
        parts = line.split()
        if len(parts) >= 12 and parts[1].lstrip('-').isdigit():
            gal_name = parts[0]
            try:
                rd_dict[gal_name] = float(parts[11])
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
                
                if eVobs <= 0: eVobs = 1.0 
                    
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

# 3. Pre-calculate Base Integrals and Geometric Matrices
print(f"Pre-calculating geometric matrices for {len(galaxies)} galaxies...")

for i, gal in enumerate(galaxies):
    print(f"Processing {gal} ({i+1}/{len(galaxies)})...", end='\r')
    
    g = galaxies[gal]
    R_m = g['R'] * kpc_to_m
    N = len(R_m)
    
    V_gas_ms = g['Vgas'] * 1000.0
    V_disk_ms = g['Vdisk'] * 1000.0
    V_bul_ms = g['Vbul'] * 1000.0
    
    hz_kpc = get_sparc_galaxy_scale_height(None, rd_dict[gal])
    hz_m = hz_kpc * kpc_to_m
    g['hz_m'] = hz_m
    
    # Calculate trapezoidal integration weights for non-uniform radial grids
    w = np.zeros(N)
    if N > 1:
        w[0] = (R_m[1] - R_m[0]) / 2.0
        w[-1] = (R_m[-1] - R_m[-2]) / 2.0
        for j in range(1, N-1):
            w[j] = (R_m[j+1] - R_m[j-1]) / 2.0
    else:
        w[0] = 1.0

    # Build the Geometric Integration Matrix M_geom for instant matrix-math later
    M_geom = np.zeros((N, N))
    for i_idx in range(N):
        for j_idx in range(N):
            def theta_integrand(theta):
                num = R_m[i_idx] - R_m[j_idx] * np.cos(theta)
                den = (R_m[i_idx]**2 + R_m[j_idx]**2 - 2 * R_m[i_idx] * R_m[j_idx] * np.cos(theta) + hz_m**2)**1.5
                return num / den
            ang_int, _ = quad(theta_integrand, 0, 2 * np.pi)
            M_geom[i_idx, j_idx] = G * R_m[i_idx] * R_m[j_idx] * ang_int * w[j_idx]
            
    g['M_geom'] = M_geom
    
    # Pre-calculate base dark mass for P=1
    valid = R_m > 0
    Sigma_b = np.zeros_like(R_m)
    V_bar_sq = np.clip(V_disk_ms**2 + V_bul_ms**2 + np.sign(V_gas_ms)*(V_gas_ms**2), 0, None)
    Sigma_b[valid] = V_bar_sq[valid] / (2 * np.pi * G * R_m[valid])
    
    rho_b = Sigma_b / (2 * hz_m)
    n_R = rho_b / m_p 
    
    rho_dm_base = (1.0 / c**3) * (n_R ** (2/3))
    Sigma_dm_base = rho_dm_base * (2 * hz_m)
    
    # Fast matrix multiplication handles the integration
    g['V_dm_sq_base_kms'] = (M_geom @ Sigma_dm_base) / (1000.0**2)
    
    # MOND Prediction
    g_N = np.zeros_like(R_m)
    g_N[valid] = V_bar_sq[valid] / R_m[valid]
    g_mond = (g_N + np.sqrt(g_N**2 + 4 * g_N * a0_mond)) / 2.0
    g['V_mond_kms'] = np.sqrt(g_mond * R_m) / 1000.0

print("\nIntegrations complete. Optimizing global parameter (Linear Absolute Error)...")

# 4. Global Optimization (L1 Norm)
def global_linear_err(P_test):
    total_linear = 0
    for gal in galaxies:
        g = galaxies[gal]
        V_dm_test_sq = P_test * g['V_dm_sq_base_kms']
        V_bar_sq_kms = np.clip(g['Vdisk']**2 + g['Vbul']**2 + np.sign(g['Vgas'])*(g['Vgas']**2), 0, None)
        V_tot_test = np.sqrt(V_bar_sq_kms + V_dm_test_sq)
        total_linear += np.sum(np.abs(g['Vobs'] - V_tot_test) / g['eVobs'])
    return total_linear

result = minimize_scalar(global_linear_err, bounds=(2.0, 10.0), method='bounded')
best_global_P = result.x

# 5. Forward Kinematics Plots
print("\nGenerating standard kinematics plots...")
all_V_obs = []
all_V_pred = []

for gal in galaxies:
    g = galaxies[gal]
    V_dm_final_sq = best_global_P * g['V_dm_sq_base_kms']
    V_dm_final_kms = np.sqrt(np.clip(V_dm_final_sq, 0, None))
    V_bar_sq_kms = np.clip(g['Vdisk']**2 + g['Vbul']**2 + np.sign(g['Vgas'])*(g['Vgas']**2), 0, None)
    V_tot_final_kms = np.sqrt(V_bar_sq_kms + V_dm_final_sq)
    
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
    plt.savefig(os.path.join(output_dir, f"{gal}_kinematics.png"), bbox_inches='tight', dpi=200)
    plt.close()

# 6. Coupled Inverse Kinematics (Solving for true V_gasp)
print("Solving coupled equations for missing baryons (V_gasp)...")

for gal in galaxies:
    g = galaxies[gal]
    R_m = g['R'] * kpc_to_m
    hz_m = g['hz_m']
    M_geom = g['M_geom']
    
    V_obs_ms_sq = (g['Vobs'] * 1000.0)**2
    V_disk_ms_sq = (g['Vdisk'] * 1000.0)**2
    V_bul_ms_sq = (g['Vbul'] * 1000.0)**2
    
    # System of equations to minimize
    def inverse_residuals(v_gasp_sq_guess):
        # 1. Total baryons (clip to prevent imaginary physics)
        V_bar_sq = np.clip(V_disk_ms_sq + V_bul_ms_sq + v_gasp_sq_guess, 0, None)
        
        # 2. Derive densities based on the guessed gas
        valid = R_m > 0
        Sigma_b = np.zeros_like(R_m)
        Sigma_b[valid] = V_bar_sq[valid] / (2 * np.pi * G * R_m[valid])
        rho_b = Sigma_b / (2 * hz_m)
        n_R = rho_b / m_p
        
        # 3. Generate corresponding Dark Mass
        rho_dm = (best_global_P / c**3) * (n_R ** (2/3))
        Sigma_dm = rho_dm * (2 * hz_m)
        
        # 4. Integrate Dark Mass gravity via the precomputed matrix
        V_dm_sq = M_geom @ Sigma_dm
        V_dm_sq = np.clip(V_dm_sq, 0, None)
        
        # 5. Total resulting velocity
        V_tot_sq = V_disk_ms_sq + V_bul_ms_sq + v_gasp_sq_guess + V_dm_sq
        
        # Goal: Difference between prediction and observation = 0
        return (V_tot_sq - V_obs_ms_sq) / 1e6  # Scaled for solver stability

    # Start the solver using the observed gas as the initial guess
    v_gasp_sq_initial = np.sign(g['Vgas']) * (g['Vgas'] * 1000.0)**2
    
    # Numerically solve the coupled system
    res = least_squares(inverse_residuals, v_gasp_sq_initial, method='lm')
    v_gasp_sq_opt = res.x
    
    # Convert solved velocity squared back to km/s (preserving sign for mass deficits)
    V_gasp_kms = np.sign(v_gasp_sq_opt) * np.sqrt(np.abs(v_gasp_sq_opt)) / 1000.0
    
    plt.figure(figsize=(8, 5))
    plt.plot(g['R'], g['Vgas'], 'b--', linewidth=2, label='V_gas (Observed)')
    plt.plot(g['R'], V_gasp_kms, 'r-', linewidth=2, label='V_gasp (Required)')
    plt.errorbar(g['R'], g['Vobs'], yerr=g['eVobs'], fmt='ko', label='V_obs (Total Target)', capsize=3)
    
    plt.axhline(0, color='gray', linestyle=':', linewidth=1)
    plt.xlabel('Radius (kpc)')
    plt.ylabel('Velocity (km/s)')
    plt.title(f'{gal} Missing Baryons (V_gasp)\nCoupled Dark Mass Solver (P = {best_global_P:.2f} W)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(inverse_dir, f"{gal}_missing_baryons.png"), bbox_inches='tight', dpi=200)
    plt.close()

# Master Scatter Plot
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

print(f"All operations completed. Inverse plots saved to '/{inverse_dir}'.")