import numpy as np
import os
from scipy.integrate import quad, trapezoid
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import warnings

# Suppress integration warnings for highly dense inner rings
warnings.filterwarnings("ignore")

# A note on velocities from stars. We measure luminousity, but want mass. So we use a constant 
# Upsilon of, say 1/2 for the mass to luminousity ratio for stars. Since velocity is a sqrt affair
# then a constant Upsilon just scales things by a factor of sqrt(1/2).
# See the 2016 paper "SPARC: MASS MODELS FOR 175 DISK GALAXIES WITH SPITZER PHOTOMETRY AND ACCURATE ROTATION CURVES."
Upsilon_disk = 0.5# 0.63#0.5   # McGaugh uses 0.5 here, our best fit is 0.63
Upsilon_bulge = 0.7 # 0.7   # McGaugh uses 0.7 here, our best fit is 0.54

sqrt_Upsilon_disk = np.sqrt(Upsilon_disk)
sqrt_Upsilon_bulge = np.sqrt(Upsilon_bulge)

# Optional Power Cuts (Set to 0.0 to disable)
MIN_DENSITY_FOR_DM = 0# Absolute minimum global density threshold (particles/cm^3)
MIN_DENSITY_WITHIN_R_D = 0   # Threshold to ignore central "gas holes" within R_d (particles/cm^3)

def get_sparc_galaxy_scale_height(radius_kpc, R_d):
    """
    Calculates the vertical disk scale height (thickness) of a SPARC galaxy.
    """
    # Empirically derived scale-height relation used for SPARC mass modeling
    z_d = 0.196 * (R_d ** 0.633)
    return z_d

# 1. Setup and Parse the Data
properties_file = "./sparc/SPARC_Lelli2016c.mrt.txt"
kinematics_file = "./sparc/MassModels_Lelli2016_SigmaGas.mrt"
output_galaxies = "./sparc/dm_mass/galaxies"
output_dir = "./sparc/dm_mass"
os.makedirs(output_galaxies, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Parse the properties file to extract R_d (disk scale length) for each galaxy
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

# Parse the kinematics file
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
                Sigma_gas = float(parts[10]) #without helium. So mult by 1.33
                
                # Multiply by sqrt(Upsilon) to convert from luminousity to mass
                Vdisk = Vdisk * sqrt_Upsilon_disk
                Vbul = Vbul * sqrt_Upsilon_bulge

                # Prevent division by zero if error is listed as 0.0
                if eVobs <= 0:
                    eVobs = 1.0 
                    
                # Only include galaxies that have a known R_d from the properties file
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
                print(line)
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
    Sigma_gas_cm2 = g_data['Sigma_gas'] 
    
    # Calculate galaxy-specific scale height
    R_d = rd_dict[gal]
    hz_kpc = get_sparc_galaxy_scale_height(None, R_d)
    hz_m = hz_kpc * kpc_to_m
    
    valid = R_m > 0
    #Sigma_b = np.zeros_like(R_m)
    
    
    # this is WRONG - this - if its anything at all - is the gas density average for all gas out to R
    Sigma_gas_m2 = Sigma_gas_cm2*100.0*100.0
    n_R_gas = Sigma_gas_m2 / (2 * hz_m) # use twice the height to get full volumne.
    n_R_gas = n_R_gas*1.33 # helium, etc
    print(n_R_gas[2]/1e6) # print particles per cm^3
    
    # Convert density thresholds from cm^-3 to m^-3
    min_n_global = MIN_DENSITY_FOR_DM * 1e6
    min_n_inner = MIN_DENSITY_WITHIN_R_D * 1e6
    
    # Create mask for density cuts
    valid_dm_density = np.ones_like(n_R_gas, dtype=bool)
    
    # Apply global cutoff
    if MIN_DENSITY_FOR_DM > 0:
        valid_dm_density &= (n_R_gas >= min_n_global)
        
    # Apply inner "gas hole" cutoff (only applies within R_d)
    if MIN_DENSITY_WITHIN_R_D > 0:
        inner_mask = g_data['R'] <= R_d
        valid_dm_density[inner_mask] &= (n_R_gas[inner_mask] >= min_n_inner)

    # Calculate Base Dark Mass Profile (P=1)
    rho_dm_base = np.zeros_like(n_R_gas)
    rho_dm_base[valid_dm_density] = (1.0 / c**3) * (n_R_gas[valid_dm_density] ** (2/3))
    Sigma_dm_base = rho_dm_base * (2 * hz_m)
    
    V_dm_sq_base = np.zeros_like(R_m)
    # I'm not sure of this integral stuff the LLM made. It seems broken. 
    # so need to do this right by hand. From first principles.

    # as a test of my gas density numbers, i want to make an array of Vgas_test from the gas density
    V_gas_sq = np.zeros_like(R_m)
 
    for i_idx, R_i in enumerate(R_m):
        if R_i == 0:
            continue
            
        radial_integrand = np.zeros_like(R_m)
        radial_integrand_gas = np.zeros_like(R_m)
        for j_idx, R_j in enumerate(R_m):
            if R_j == 0:
                continue
                
            def theta_integrand(theta):
                numerator = R_i - R_j * np.cos(theta)
                denominator = (R_i**2 + R_j**2 - 2 * R_i * R_j * np.cos(theta) + hz_m**2)**1.5
                return numerator / denominator
                
            ang_int, _ = quad(theta_integrand, 0, 2 * np.pi)
            radial_integrand[j_idx] = Sigma_dm_base[j_idx] * R_j * ang_int
            radial_integrand_gas[j_idx] = Sigma_gas_m2[j_idx] * m_p * R_j * ang_int
            
        V_dm_sq_val = G * R_i * trapezoid(radial_integrand, R_m)
        V_dm_sq_base[i_idx] = max(0, V_dm_sq_val)

        V_gas_sq[i_idx] = G * R_i * trapezoid(radial_integrand_gas, R_m)
        
    g_data['V_dm_sq_base_kms'] = V_dm_sq_base / (1000.0**2)
    g_data['Vgas_test'] = np.sign(V_gas_sq)*np.sqrt(np.fabs(V_gas_sq))/1000.0
     
    # MOND Prediction Calculation
    # Estimate baryonic surface mass density (using sign preservation for gas holes)
    V_bar_sq = np.clip(V_disk_ms**2 + V_bul_ms**2 + V_gas_ms * np.abs(V_gas_ms), 0, None)

    g_N = np.zeros_like(R_m)
    g_N[valid] = V_bar_sq[valid] / R_m[valid]
    
    # Simple MOND interpolation function
    g_mond = (g_N + np.sqrt(g_N**2 + 4 * g_N * a0_mond)) / 2.0
    g_data['V_mond_kms'] = np.sqrt(g_mond * R_m) / 1000.0

print("\nIntegrations complete. Optimizing global parameter for Linear Absolute Error...")

# 4. Global Optimization Function (Minimizing Linear Error)
def global_linear_err(P_test):
    total_linear = 0
    for gal in galaxies:
        g = galaxies[gal]
        V_dm_test_sq = P_test * g['V_dm_sq_base_kms']
        
        V_bar_sq_kms = np.clip(g['Vdisk']**2 + g['Vbul']**2 + g['Vgas'] * np.abs(g['Vgas']), 0, None)
        V_tot_test = np.sqrt(V_bar_sq_kms + V_dm_test_sq)
        
        # Absolute Linear Error (L1 Norm)
        linear_err = np.sum(np.abs(g['Vobs'] - V_tot_test) / g['eVobs'])
        #linear_err = np.sum(np.abs(g['Vobs'] - V_tot_test)) # this gets error in km/s squared, divide by number of points to get km/sec
        total_linear += linear_err
        
    return total_linear

result = minimize_scalar(global_linear_err, bounds=(2.0, 100.0), method='bounded')
best_global_P = result.x
min_linear_err = result.fun

# Calculate the resulting Chi2 for Dark Mass, and both errors for MOND
dm_chi2_total = 0
mond_linear_total = 0
mond_chi2_total = 0

for gal in galaxies:
    g = galaxies[gal]
    
    # Dark Mass Finals
    V_dm_final_kms = np.sqrt(best_global_P * g['V_dm_sq_base_kms'])
    V_bar_sq_kms = np.clip(g['Vdisk']**2 + g['Vbul']**2 + g['Vgas'] * np.abs(g['Vgas']), 0, None)
    V_tot_final_kms = np.sqrt(V_bar_sq_kms + V_dm_final_kms**2)
    
    dm_chi2_total += np.sum(((g['Vobs'] - V_tot_final_kms) / g['eVobs'])**2)
    
    # MOND Finals
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
    
    # Save with doubled linear resolution
    plt.savefig(os.path.join(output_galaxies, f"{gal}_dm_mass.png"), bbox_inches='tight', dpi=300)
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

plt.savefig(os.path.join(output_dir, "global_fit_vs_obs.png"), bbox_inches='tight', dpi=300)
plt.close()

# --- Generate Acceleration vs Acceleration Scatter Plot ---
print("\nGenerating global acceleration scatter plot...")

all_g_obs = []
all_g_pred = []

for gal in galaxies:
    g = galaxies[gal]
    R_m = g['R'] * kpc_to_m
    valid = R_m > 0  # Avoid division by zero
    
    # Calculate final dark mass velocity squared
    V_dm_final_sq = best_global_P * g['V_dm_sq_base_kms']
    V_bar_sq_kms = np.clip(g['Vdisk']**2 + g['Vbul']**2 + np.sign(g['Vgas'])*(g['Vgas']**2), 0, None)
    
    # Total predicted V^2 in (m/s)^2
    V_tot_final_sq_ms = (V_bar_sq_kms + V_dm_final_sq) * (1000.0**2)
    V_obs_sq_ms = (g['Vobs'] * 1000.0)**2
    
    # Convert V^2 to centripetal acceleration: g = V^2 / R
    g_obs = V_obs_sq_ms[valid] / R_m[valid]
    g_pred = V_tot_final_sq_ms[valid] / R_m[valid]
    
    all_g_obs.extend(g_obs)
    all_g_pred.extend(g_pred)

plt.figure(figsize=(9, 9))
plt.scatter(all_g_obs, all_g_pred, alpha=0.3, edgecolors='none', c='blue')

# Set Logarithmic scales
plt.xscale('log')
plt.yscale('log')

# Dynamically find the clean powers of 10 for the axis bounds
min_val = 10**np.floor(np.log10(min(min(all_g_obs), min(all_g_pred))))
max_val = 10**np.ceil(np.log10(max(max(all_g_obs), max(all_g_pred))))

# 45-degree Perfect Fit / Newton Line
plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Newton')

plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)
plt.axis('square')  # Lock aspect ratio to be perfectly square

plt.xlabel('Observed Acceleration $g_{obs}$ (m/s$^2$)', fontsize=12)
plt.ylabel('Predicted Acceleration $g_{pred}$ (m/s$^2$)', fontsize=12)
plt.title(f'Global Fit: Dark Mass vs Observed Acceleration\n(Total Points = {len(all_g_obs)}, Universal P = {best_global_P:.2f} W)', fontsize=14)
plt.legend()

# Add a finer grid for log scales
plt.grid(True, which="both", ls="--", alpha=0.5)

plt.savefig(os.path.join(output_dir, "global_predicted_accel_vs_obs.png"), bbox_inches='tight', dpi=200)
plt.close()


print(f"All operations completed successfully. Files saved to '/{output_dir}'.")