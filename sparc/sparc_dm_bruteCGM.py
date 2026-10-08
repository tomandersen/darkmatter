import numpy as np
import os
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import warnings

# Suppress integration warnings for highly dense inner rings
warnings.filterwarnings("ignore")

Upsilon_disk = 0.5   
Upsilon_bulge = 0.7  

sqrt_Upsilon_disk = np.sqrt(Upsilon_disk)
sqrt_Upsilon_bulge = np.sqrt(Upsilon_bulge)
 
NUM_BLOBS_THETA = 57 
NUM_LAYERS_Z = 303
Z_HEIGHT = 100  

NUM_R_BINS = 87  
USE_EXPONENTIAL_DISK = True

USE_CGM_MODEL = False 
GGM_GRID_SCALE = 2.0
CGM_CENTRAL_DENSITY = 0.3

def get_sparc_galaxy_scale_height(radius_kpc, R_d):
    """Calculates the vertical disk scale height (thickness) of a SPARC galaxy."""
    z_d = 0.196 * (R_d ** 0.633)
    return z_d

# 1. Setup and Parse the Data
properties_file = "./sparc/SPARC_Lelli2016c.mrt.txt"
kinematics_file = "./sparc/MassModels_Lelli2016_SigmaGas.mrt"
output_galaxies = "./sparc/dm_mass/galaxies"
output_dir = "./sparc/dm_mass"
output_gas_heatmaps = "./sparc/dm_mass/gas_heatmaps"
output_dm_heatmaps = "./sparc/dm_mass/dm_heatmaps"

os.makedirs(output_galaxies, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_gas_heatmaps, exist_ok=True)
os.makedirs(output_dm_heatmaps, exist_ok=True)

rd_dict = {}
HIMass_dict = {}
TL_dict = {}
with open(properties_file, 'r') as f:
    for line in f:
        parts = line.split()
        if len(parts) >= 12 and parts[1].lstrip('-').isdigit():
            gal_name = parts[0]
            try:
                r_d = float(parts[11])
                rd_dict[gal_name] = r_d
                t_luminosity = float(parts[7])
                TL_dict[gal_name] = t_luminosity
                hi_mass = float(parts[13])
                HIMass_dict[gal_name] = hi_mass
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
                Sigma_gas = float(parts[10]) 
                
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
print(f"Pre-calculating base arrays for {len(galaxies)} galaxies via continuous 3D brute force...")

for i, gal in enumerate(galaxies):
    print(f"Processing {gal} ({i+1}/{len(galaxies)})...", end='\r')
    
    g_data = galaxies[gal]
    R_obs_kpc = g_data['R']
    R_m_obs = R_obs_kpc * kpc_to_m
    V_gas_ms = g_data['Vgas'] * 1000.0
    V_disk_ms = g_data['Vdisk'] * 1000.0
    V_bul_ms = g_data['Vbul'] * 1000.0
    
    max_R_kpc = np.max(R_obs_kpc)
    grid_max = max_R_kpc
    if USE_CGM_MODEL:
        grid_max = GGM_GRID_SCALE * grid_max

    R_grid_kpc = np.linspace(max_R_kpc / NUM_R_BINS, grid_max, NUM_R_BINS)
    R_grid_m = R_grid_kpc * kpc_to_m
    dR_m = R_grid_m[1] - R_grid_m[0]
    
    g_data_r_obs_kpc = R_obs_kpc
    g_data_sigma_gas = g_data['Sigma_gas']
    g_data_Vdisk = g_data['Vdisk']
    g_data_Vbul = g_data['Vbul']
    g_data_Vgas = g_data['Vgas']
    
    if USE_CGM_MODEL: 
        rscale = 1.0/(GGM_GRID_SCALE * GGM_GRID_SCALE)
        g_data_r_obs_kpc = np.append(R_obs_kpc, grid_max)
        g_data_sigma_gas = np.append(g_data_sigma_gas, 0.0)
        g_data_Vdisk = np.append(g_data_Vdisk, rscale*g_data_Vdisk[-1])
        g_data_Vbul = np.append(g_data_Vbul, rscale*g_data_Vbul[-1])
        g_data_Vgas = np.append(g_data_Vgas, rscale*g_data_Vgas[-1])
    
    Sigma_gas_cm2_grid = np.interp(R_grid_kpc, g_data_r_obs_kpc, g_data_sigma_gas)
    g_data['Vdisk_grid'] = np.interp(R_grid_kpc, g_data_r_obs_kpc, g_data_Vdisk)
    g_data['Vbul_grid'] = np.interp(R_grid_kpc, g_data_r_obs_kpc, g_data_Vbul)
    g_data['Vgas_grid'] = np.interp(R_grid_kpc, g_data_r_obs_kpc, g_data_Vgas)
    
    R_d = rd_dict[gal]
    hz_kpc = get_sparc_galaxy_scale_height(None, R_d)
    hz_m = hz_kpc * kpc_to_m
    
    valid_obs = R_m_obs > 0 
    Sigma_gas_m2_grid = Sigma_gas_cm2_grid * 100.0 * 100.0 
    
    # ---------------------------------------------------------
    # BRUTE FORCE 3D BLOB CALCULATION (Continuous Grid)
    # ---------------------------------------------------------
    nz = NUM_LAYERS_Z
    ntheta = NUM_BLOBS_THETA
    
    Z_m = np.linspace(-Z_HEIGHT * hz_m, Z_HEIGHT * hz_m, nz) 
    dZ = Z_m[1] - Z_m[0] if nz > 1 else 2.0 * hz_m 
    
    theta_rad = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
    dTheta = 2 * np.pi / ntheta

    R_grid_3d, Z_grid_3d, Theta_grid_3d = np.meshgrid(R_grid_m, Z_m, theta_rad, indexing='ij')
    
    dV = R_grid_3d * dTheta * dR_m * dZ 
    
    gas_particle_density_1d = np.where(R_grid_m > 0, Sigma_gas_m2_grid / (2.0 * hz_m), 0.0)
    gas_particle_density_grid = gas_particle_density_1d[:, None, None] * np.ones_like(Z_grid_3d)

    if USE_EXPONENTIAL_DISK:
        gas_particle_density_grid *= 0.5*np.exp(-np.abs(Z_grid_3d) / hz_m)
    else: 
        z_mask = np.abs(Z_grid_3d) <= hz_m
        gas_particle_density_grid[~z_mask] = 0.0

    X_blob = R_grid_3d * np.cos(Theta_grid_3d)
    Y_blob = R_grid_3d * np.sin(Theta_grid_3d)
    Z_blob = Z_grid_3d

    if USE_CGM_MODEL:
        total_mass = TL_dict[gal] + HIMass_dict[gal] 

        r_core_kpc = 2.0*(total_mass*1e9/1e11)**(1/3)
        r_core_m = r_core_kpc*kpc_to_m
        n_0_cgs = CGM_CENTRAL_DENSITY  
        n_0_m3 =  n_0_cgs * (100)**3
        beta = 2.0/3.0
        exponent = -3.0*beta/2.0
        gas_particle_density_grid += n_0_m3*(1 + (X_blob**2 + Y_blob**2 + Z_blob**2)/r_core_m**2)**exponent

    # --- NEW: Attach 2D slice at y=0 (theta=0) for heatmaps ---
    g_data['R_grid_kpc'] = R_grid_kpc
    g_data['Z_grid_kpc'] = Z_m / kpc_to_m
    g_data['gas_density_slice_m3'] = gas_particle_density_grid[:, :, 0].copy()

    gas_particle_mass_He_factor = 1.33  
    dm_gas = gas_particle_density_grid *m_p*gas_particle_mass_He_factor * dV
    dm_base_dm = (1/c**3)*gas_particle_density_grid**(2/3) * dV 

    V_gas_sq_grid = np.zeros_like(R_grid_m)
    V_dm_sq_base_grid = np.zeros_like(R_grid_m)
    
    softening_sq = (0.01 * hz_m)**2 
    
    for i_idx, R_test in enumerate(R_grid_m):
        dx = X_blob - R_test
        dy = Y_blob
        dz = Z_blob
        
        dist_sq = dx**2 + dy**2 + dz**2 + softening_sq
        dist = np.sqrt(dist_sq)
        
        da_x_factor = G * dx / (dist_sq * dist)
        
        a_x_gas = np.sum(da_x_factor * dm_gas)
        a_x_dm = np.sum(da_x_factor * dm_base_dm)
        
        V_gas_sq_grid[i_idx] = -a_x_gas * R_test
        V_dm_sq_base_grid[i_idx] = max(0, -a_x_dm * R_test)
        
    g_data['V_dm_sq_base_grid_kms'] = V_dm_sq_base_grid / 1e6
    g_data['Vgas_test_grid_kms'] = np.sign(V_gas_sq_grid) * np.sqrt(np.abs(V_gas_sq_grid)) / 1000.0
    
    g_data['V_dm_sq_base_kms'] = np.interp(R_obs_kpc, R_grid_kpc, V_dm_sq_base_grid / 1e6)
    g_data['Vgas_test'] = np.interp(R_obs_kpc, R_grid_kpc, g_data['Vgas_test_grid_kms'])
    
    V_bar_sq = np.clip(V_disk_ms**2 + V_bul_ms**2 + V_gas_ms * np.abs(V_gas_ms), 0, None)
    g_N = np.zeros_like(R_m_obs)
    g_N[valid_obs] = V_bar_sq[valid_obs] / R_m_obs[valid_obs]
    
    g_mond = (g_N + np.sqrt(g_N**2 + 4 * g_N * a0_mond)) / 2.0
    g_data['V_mond_kms'] = np.sqrt(g_mond * R_m_obs) / 1000.0


print("\nIntegrations complete. Optimizing global parameter for Linear Absolute Error...")


# 4. Global Optimization Function
def global_fit_err(P_test):
    total_err = 0
    for gal in galaxies:
        g = galaxies[gal]
        V_dm_test_sq = P_test * g['V_dm_sq_base_kms']
        
        V_bar_sq_kms = np.clip(g['Vdisk']**2 + g['Vbul']**2 + g['Vgas'] * np.abs(g['Vgas']), 0, None)
        V_tot_test = np.sqrt(V_bar_sq_kms + V_dm_test_sq)
        
        the_err = np.sum(((g['Vobs'] - V_tot_test) / g['eVobs'])**2) 
        total_err += the_err

    print(f"Current P_test: {P_test:.4f} Watts, Current Error: {total_err:.2f}")   
    return total_err

result = minimize_scalar(global_fit_err, bounds=(2.0, 100.0), method='bounded')
best_global_P = result.x
min_found_err = result.fun
    
dm_chi2_total = 0
dm_linear_total = 0
dm_linear_total_err = 0
mond_chi2_total = 0
mond_linear_total = 0
mond_linear_total_err = 0

for gal in galaxies:
    g = galaxies[gal]
    V_dm_final_kms = np.sqrt(best_global_P * g['V_dm_sq_base_kms'])
    V_bar_sq_kms = np.clip(g['Vdisk']**2 + g['Vbul']**2 + g['Vgas'] * np.abs(g['Vgas']), 0, None)
    V_tot_final_kms = np.sqrt(V_bar_sq_kms + V_dm_final_kms**2)
    
    dm_chi2_total += np.sum(((g['Vobs'] - V_tot_final_kms) / g['eVobs'])**2)
    dm_linear_total_err += np.sum(np.abs(g['Vobs'] - V_tot_final_kms) / g['eVobs'])
    dm_linear_total += np.sum(np.abs(g['Vobs'] - V_tot_final_kms)) 

    mond_chi2_total += np.sum(((g['Vobs'] - g['V_mond_kms']) / g['eVobs'])**2)
    mond_linear_total_err += np.sum(np.abs(g['Vobs'] - g['V_mond_kms']) / g['eVobs'])
    mond_linear_total += np.sum(np.abs(g['Vobs'] - g['V_mond_kms'])) 


# ---------------------------------------------------------
# NEW FUNCTION: Generate and output heatmaps
# ---------------------------------------------------------
def generate_density_heatmaps(galaxy_dict, best_P):
    print("\nGenerating 2D Gas and DM Heatmaps...")
    for gal in galaxy_dict:
        g = galaxy_dict[gal]
        R_kpc = g['R_grid_kpc']
        Z_kpc = g['Z_grid_kpc']
        gas_density_m3 = g['gas_density_slice_m3']
        
        # 1) Gas density to particles/cm^3 (1 m^3 = 10^6 cm^3)
        gas_density_cm3 = gas_density_m3 / 1e6
        
        # 2) DM density (mass density in kg/m^3) -> particles/cm^3 (divide by m_p and 10^6)
        dm_density_kg_m3 = best_P * (1/c**3) * (gas_density_m3**(2/3))
        dm_density_cm3 = dm_density_kg_m3 / (m_p * 1e6)
        
        R_mesh, Z_mesh = np.meshgrid(R_kpc, Z_kpc, indexing='ij')

        # --- Plot Gas Heatmap ---
        plt.figure(figsize=(10, 6))
        # Ensure log scale works by adding a tiny epsilon if 0 exists
        vmin = max(gas_density_cm3.max() * 1e-4, 1e-6)
        pcm = plt.pcolormesh(R_mesh, Z_mesh, gas_density_cm3, 
                             shading='auto', cmap='viridis', 
                             norm=mcolors.LogNorm(vmin=vmin, vmax=gas_density_cm3.max()))
        plt.colorbar(pcm, label='Gas Density (particles / cm$^3$)')
        plt.xlabel('Radius [x-axis] (kpc)')
        plt.ylabel('Height [z-axis] (kpc)')
        plt.title(f'{gal} 2D Gas Density (y=0 slice)')
        plt.savefig(os.path.join(output_gas_heatmaps, f"{gal}_gas_heatmap.png"), bbox_inches='tight', dpi=200)
        plt.close()

        # --- Plot DM Heatmap ---
        plt.figure(figsize=(10, 6))
        vmin_dm = max(dm_density_cm3.max() * 1e-4, 1e-6)
        pcm = plt.pcolormesh(R_mesh, Z_mesh, dm_density_cm3, 
                             shading='auto', cmap='plasma', 
                             norm=mcolors.LogNorm(vmin=vmin_dm, vmax=dm_density_cm3.max()))
        plt.colorbar(pcm, label='DM Density Equivalent (protons / cm$^3$)')
        plt.xlabel('Radius [x-axis] (kpc)')
        plt.ylabel('Height [z-axis] (kpc)')
        plt.title(f'{gal} 2D Dark Mass Density (y=0 slice)\n$P={best_P:.2f}W$')
        plt.savefig(os.path.join(output_dm_heatmaps, f"{gal}_dm_heatmap.png"), bbox_inches='tight', dpi=200)
        plt.close()

generate_density_heatmaps(galaxies, best_global_P)

# 5. Generate Output Plots
print("\nGenerating individual galaxy plots and master summary...")
all_V_obs = []
all_V_pred = []

for gal in galaxies:
    g = galaxies[gal]
    
    V_dm_final_obs_kms = np.sqrt(best_global_P * g['V_dm_sq_base_kms'])
    V_bar_sq_obs_kms = np.clip(g['Vdisk']**2 + g['Vbul']**2 + g['Vgas'] * np.abs(g['Vgas']), 0, None)
    V_tot_final_obs_kms = np.sqrt(V_bar_sq_obs_kms + V_dm_final_obs_kms**2)
    
    all_V_obs.extend(g['Vobs'])
    all_V_pred.extend(V_tot_final_obs_kms)
    
    V_dm_final_grid_kms = np.sqrt(best_global_P * g['V_dm_sq_base_grid_kms'])
    V_bar_sq_grid_kms = np.clip(g['Vdisk_grid']**2 + g['Vbul_grid']**2 + g['Vgas_grid'] * np.abs(g['Vgas_grid']), 0, None)
    V_tot_final_grid_kms = np.sqrt(V_bar_sq_grid_kms + V_dm_final_grid_kms**2)
    
    plt.figure(figsize=(9, 6))

    plt.plot(g['R_grid_kpc'], V_dm_final_grid_kms, 'm--', label=f'dark mass (dm)')
    plt.plot(g['R_grid_kpc'], V_tot_final_grid_kms, 'r-', linewidth=2, label='baryons + dm')

    plt.plot(g['R'], g['Vgas'], 'g', label='gas', linestyle='dotted', linewidth=0.7)
    plt.plot(g['R'], g['Vdisk'], 'violet', label='disk', linestyle='dashdot', linewidth=0.7)
    if np.any(g['Vbul'] > 0):
        plt.plot(g['R'], g['Vbul'], 'red', label='bulge', linestyle='dashed', linewidth=0.7)
  
    plt.plot(g['R_grid_kpc'], g['Vgas_test_grid_kms'], 'g', label="gas ($\\Sigma_{gas}$)", linewidth=0.5)
    plt.plot(g['R_grid_kpc'], np.sqrt(V_bar_sq_grid_kms), 'blue', label='all baryons', linewidth=0.7)
    
    max_r = max(g['R'])
    plt.xlim(0.0, max_r) 
    plt.plot(g['R'], g['V_mond_kms'], 'c-.', linewidth=1, label='MOND')
 
    plt.errorbar(g['R'], g['Vobs'], yerr=g['eVobs'], fmt='ko', label='observed', capsize=1.5, capthick=1.0, markersize=2, elinewidth=0.6)
    
    plt.xlabel('Radius (kpc)')
    plt.ylabel('Velocity (km/s)')
    plt.title(f'{gal} Kinematics (Universal P = {best_global_P:.2f} W)')
    leg = plt.legend(fontsize="small")
    leg.get_frame().set_linewidth(0.0) 

    plt.grid(True, linewidth=0.5, alpha=0.7)
    
    plt.savefig(os.path.join(output_galaxies, f"{gal}_dm_mass.png"), bbox_inches='tight', dpi=300)
    plt.close()

plt.figure(figsize=(9, 9))
plt.scatter(all_V_obs, all_V_pred, alpha=0.3, edgecolors='none', c='blue')
max_val = max(max(all_V_obs), max(all_V_pred))
plt.plot([0, max_val], [0, max_val], 'k--', linewidth=2, label='Perfect Fit (1:1)')

plt.xlabel('Observed Velocity (km/s)', fontsize=12)
plt.ylabel('Predicted Velocity (km/s)', fontsize=12)
plt.title(f'Global Fit: Dark Mass Theory vs SPARC Dataset\n(Total Points = {len(all_V_obs)}, Universal P = {best_global_P:.2f} W)', fontsize=14)
plt.legend(fontsize="small")
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
plt.scatter(all_g_pred, all_g_obs, alpha=0.3, edgecolors='none', c='blue')

plt.xscale('log')
plt.yscale('log')

min_val = 10**np.floor(np.log10(min(min(all_g_obs), min(all_g_pred))))
max_val = 10**np.ceil(np.log10(max(max(all_g_obs), max(all_g_pred))))

plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Newton')

plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)
plt.axis('square')

plt.ylabel('Observed Acceleration $g_{obs}$ (m/s$^2$)', fontsize=12)
plt.xlabel('Predicted Acceleration $g_{pred}$ (m/s$^2$)', fontsize=12)
plt.title(f'Global Fit: Dark Mass vs Observed Acceleration\n(Total Points = {len(all_g_obs)}, Universal P = {best_global_P:.2f} W)', fontsize=14)
plt.legend(fontsize="small")
plt.grid(True, which="both", ls="--", alpha=0.5)

plt.savefig(os.path.join(output_dir, "global_predicted_accel_vs_obs.png"), bbox_inches='tight', dpi=200)
plt.close()

print(f"All operations completed successfully. Files saved to '/{output_dir}'.")