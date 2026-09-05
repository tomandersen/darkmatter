import numpy as np
from scipy.spatial import cKDTree

# Following the gemini app discussion "Milky Way Mock Catalogs"
# download (not included in git repo the m12i simulation)
# https://users.flatironinstitute.org/~mgrudic/fire2_public_release/core/m12i_res7100/output/snapdir_600/ 
# (it says you need to be a member of the FIRE collaboration to access this, but it worked for me just fine)


import numpy as np
import matplotlib.pyplot as plt
import gizmo_analysis as gizmo

# ==========================================
# 1. CONFIGURATION & CONSTANTS
# ==========================================
SNAPSHOT_NUM = 600
DENSITY_THRESHOLD = 0.0001  # cm^-3 (filtering for diffuse gas)
DISTANCE_LIMIT = 30.0    # kpc

G = 4.3009e-6             # Gravitational constant: kpc * (km/s)^2 / M_sun
c = 299792458.0           # Speed of light: m/s
MSUN_TO_KG = 1.989e30     # Solar masses to kilograms
MSUN_KPC3_TO_KG_M3 = 6.77e-29 # Mass density conversion

def get_component_velocity(radii, masses):
    """Sorts particles by radius and calculates circular velocity."""
    sort_idx = np.argsort(radii)
    r_sorted = radii[sort_idx]
    m_sorted = masses[sort_idx]
    
    enclosed_mass = np.cumsum(m_sorted)
    v_circ = np.sqrt((G * enclosed_mass) / (r_sorted + 1e-6)) # Avoid div by zero
    return r_sorted, v_circ

# ==========================================
# 2. LOAD DATA & EXTRACT PROPERTIES
# ==========================================
print("Loading snapshot data...")

part = gizmo.io.Read.read_snapshots(
    ['star', 'gas', 'dark'], 
    'index', 
    SNAPSHOT_NUM, 
    assign_hosts=True,
    simulation_directory='./m10q_res30'
)

# Extract Radii
# Changed from 'host.distance.principal.spherical' to 'host.distance.spherical'
r_stars = part['star'].prop('host.distance.spherical')[:, 0]
r_gas_all = part['gas'].prop('host.distance.spherical')[:, 0]
r_dm_all = part['dark'].prop('host.distance.spherical')[:, 0]

# star positions
pos_stars_all = part['star'].prop('host.distance') # 3D Cartesian coords

# Extract Masses
mass_stars = part['star']['mass']
mass_gas_all = part['gas']['mass']
mass_dm_all = part['dark']['mass']

# ==========================================
# 3. FILTER GAS & TRUE DARK MATTER
# ==========================================
print("Filtering for low-density gas and isolating 100 kpc volume...")
gas_density_cm3 = part['gas'].prop('number.density') 

# Gas filter mask
low_density_mask = (gas_density_cm3 < DENSITY_THRESHOLD) & (r_gas_all < DISTANCE_LIMIT)
r_gas = r_gas_all[low_density_mask]
mass_gas = mass_gas_all[low_density_mask]
n_cm3 = gas_density_cm3[low_density_mask]
gas_rho_msun_kpc3 = part['gas'].prop('density')[low_density_mask]

# Star filter mask (to save RAM)
star_mask = r_stars < DISTANCE_LIMIT
r_stars_f = r_stars[star_mask]
mass_stars_f = mass_stars[star_mask]
pos_stars_f = pos_stars_all[star_mask]

# True DM filter mask (to calculate target total DM mass)
dm_mask = r_dm_all < DISTANCE_LIMIT
r_dm_true = r_dm_all[dm_mask]
mass_dm_true = mass_dm_all[dm_mask]

# FIX: Force np.sum to use 64-bit precision before multiplying by 10^30
target_dm_mass_kg = np.sum(mass_dm_true, dtype=np.float64) * MSUN_TO_KG

# ==========================================
# 4. RUN YOUR DARK MATTER MODEL (GAS + STARS)
# ==========================================
print("Running DM model for Gas...")
n_m3_gas = n_cm3.astype(np.float64) * 1e6
gas_mass_kg = mass_gas.astype(np.float64) * MSUN_TO_KG
gas_rho_kg_m3 = gas_rho_msun_kpc3.astype(np.float64) * MSUN_KPC3_TO_KG_M3

V_i_gas = gas_mass_kg / gas_rho_kg_m3
d_avg_gas = np.cbrt(1.0 / n_m3_gas)

print("Calculating local density for Stars using KD-Tree...")
# 1. Build a spatial tree and find the 16th nearest neighbor for every star
tree = cKDTree(pos_stars_f)
# k=17 because the 1st neighbor is the particle itself
distances, _ = tree.query(pos_stars_f, k=17, workers=-1) 
r_16 = distances[:, 16].astype(np.float64) # Distance in kpc

# 2. Calculate Stellar Volume and Density
V_sphere_kpc3 = (4.0 / 3.0) * np.pi * (r_16**3)
# Local mass density in kg/m^3
star_rho_kg_m3 = ((16.0 * mass_stars_f.mean() * MSUN_TO_KG) / V_sphere_kpc3) * MSUN_KPC3_TO_KG_M3
V_i_star = (mass_stars_f.astype(np.float64) * MSUN_TO_KG) / star_rho_kg_m3

# 3. Convert stellar mass density to baryon number density
# Average mass of a baryon (proton/neutron) is ~1.67e-27 kg
mass_baryon_kg = 1.67e-27 
n_m3_star = star_rho_kg_m3 / mass_baryon_kg
d_avg_star = np.cbrt(1.0 / n_m3_star)

print("Calculating global Power P...")
alpha_std = 2.0

# Sum the contributions from both Gas and Stars
sum_factor_gas = np.sum(V_i_gas / (c**3 * d_avg_gas**alpha_std))
sum_factor_star = np.sum(V_i_star / (c**3 * d_avg_star**alpha_std))
sum_factor_total = sum_factor_gas + sum_factor_star

# Calculate P and distribute the predicted Dark Matter
P = target_dm_mass_kg / sum_factor_total
print(f"Calculated Shared Power P: {P:.2e} Watts")

dm_mass_msun_pred_gas = (P * (V_i_gas / (c**3 * d_avg_gas**alpha_std))) / MSUN_TO_KG
dm_mass_msun_pred_star = (P * (V_i_star / (c**3 * d_avg_star**alpha_std))) / MSUN_TO_KG

# ==========================================
# 5. CALCULATE ROTATION CURVES
# ==========================================
print("Calculating velocity curves...")
# Individual components
_, v_stars = get_component_velocity(r_stars, mass_stars)
_, v_gas = get_component_velocity(r_gas, mass_gas)
r_dm_t, v_dm_true = get_component_velocity(r_dm_true, mass_dm_true)
r_dm_p, v_dm_pred = get_component_velocity(r_gas, dm_mass_msun_pred_gas)

# Total curves (combining components)
all_r_true = np.concatenate([r_stars, r_gas, r_dm_true])
all_m_true = np.concatenate([mass_stars, mass_gas, mass_dm_true])
r_tot_true, v_tot_true = get_component_velocity(all_r_true, all_m_true)

# We use r_gas and r_stars_f twice because both are now generating DM
all_r_pred = np.concatenate([r_stars_f, r_gas, r_stars_f, r_gas]) 
all_m_pred = np.concatenate([mass_stars_f, mass_gas, dm_mass_msun_pred_star, dm_mass_msun_pred_gas])
r_tot_pred, v_tot_pred = get_component_velocity(all_r_pred, all_m_pred)
# ==========================================
# 6. PLOTTING AND SAVING THE RESULTS
# ==========================================
print("Generating and saving graphs...")

# STEP: Plot every 1,000th point to save memory. 
S = 1000 

# ------------------------------------------
# Graph 1: Total Rotation Curve
# ------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(r_tot_true[::S], v_tot_true[::S], label='True FIRE Simulation', color='black', lw=2)
plt.plot(r_tot_pred[::S], v_tot_pred[::S], label='Your Model (Baryons + Pred DM)', color='crimson', ls='--', lw=2)

plt.title('Total Rotation Curve Comparison', fontsize=14)
plt.xlabel('Galactocentric Radius (kpc)', fontsize=12)
plt.ylabel('Velocity (km/s)', fontsize=12)
plt.xlim(0, 100)
plt.ylim(0, 300)
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right')

plt.tight_layout()
plt.savefig('plot_1_total_rotation_curve.png', dpi=300)
plt.close() # Closes the figure to free up your memory!
print("Saved plot_1_total_rotation_curve.png")

# ------------------------------------------
# Graph 2: Component Decomposition
# ------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(r_dm_t[::S], v_dm_true[::S], label='True DM Halo', color='black', lw=2)
plt.plot(r_dm_p[::S], v_dm_pred[::S], label=f'Predicted DM Halo (α={alpha_std})', color='crimson', ls='--', lw=2)
plt.plot(np.sort(r_stars)[::S], v_stars[::S], label='Stars', color='goldenrod', ls='-.')
plt.plot(np.sort(r_gas)[::S], v_gas[::S], label='Gas', color='teal', ls='-.')

plt.title('Velocity Contributions by Component', fontsize=14)
plt.xlabel('Galactocentric Radius (kpc)', fontsize=12)
plt.ylabel('Velocity Contribution (km/s)', fontsize=12)
plt.xlim(0, 100)
plt.ylim(0, 350)
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right')

plt.tight_layout()
plt.savefig('plot_2_component_decomposition.png', dpi=300)
plt.close()
print("Saved plot_2_component_decomposition.png")

# ------------------------------------------
# Graph 3: Tuning the Power-Law (Alpha)
# ------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(r_dm_t[::S], v_dm_true[::S], label='True DM Halo', color='black', lw=2)
alpha_test_values = [1.0, 1.5, 2.0, 2.5]

for a in alpha_test_values:
    # 1. Sum factors for both components
    sf_gas = np.sum(V_i_gas / (c**3 * d_avg_gas**a))
    sf_star = np.sum(V_i_star / (c**3 * d_avg_star**a))
    sf_total = sf_gas + sf_star
    
    # 2. Recalculate P for this alpha
    P_a = target_dm_mass_kg / sf_total
    
    # 3. Calculate DM mass for both components
    dm_m_gas = (P_a * (V_i_gas / (c**3 * d_avg_gas**a))) / MSUN_TO_KG
    dm_m_star = (P_a * (V_i_star / (c**3 * d_avg_star**a))) / MSUN_TO_KG
    
    # 4. Combine and calculate velocity
    r_comb = np.concatenate([r_gas, r_stars_f])
    m_comb = np.concatenate([dm_m_gas, dm_m_star])
    r_a, v_a = get_component_velocity(r_comb, m_comb)
    
    # 5. Plot the result
    plt.plot(r_a[::S], v_a[::S], label=f'Model (α={a})', ls='--')

plt.title('Tuning the Distance Exponent (α)', fontsize=14)
plt.xlabel('Galactocentric Radius (kpc)', fontsize=12)
plt.ylabel('DM Velocity Contribution (km/s)', fontsize=12)
plt.xlim(0, 100)
plt.ylim(0, 350)
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right')

plt.tight_layout()
plt.savefig('plot_3_tuning_alpha.png', dpi=300)
plt.close()
print("Saved plot_3_tuning_alpha.png")