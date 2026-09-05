import numpy as np

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
DENSITY_THRESHOLD = 0.01  # cm^-3 (filtering for diffuse gas)
DISTANCE_LIMIT = 100.0    # kpc

G = 4.3009e-6             # Gravitational constant: kpc * (km/s)^2 / M_sun
c = 299792458.0           # Speed of light: m/s
MSUN_TO_KG = 1.989e30     # Solar masses to kilograms
MSUN_KPC3_TO_KG_M3 = 6.77e-20 # Mass density conversion

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
    simulation_directory='./m12i_res7100'
)

# Extract Radii
# Changed from 'host.distance.principal.spherical' to 'host.distance.spherical'
r_stars = part['star'].prop('host.distance.spherical')[:, 0]
r_gas_all = part['gas'].prop('host.distance.spherical')[:, 0]
r_dm_all = part['dark'].prop('host.distance.spherical')[:, 0]


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

# True DM filter mask (to calculate target total DM mass)
dm_mask = r_dm_all < DISTANCE_LIMIT
r_dm_true = r_dm_all[dm_mask]
mass_dm_true = mass_dm_all[dm_mask]
target_dm_mass_kg = np.sum(mass_dm_true) * MSUN_TO_KG

# ==========================================
# 4. RUN YOUR DARK MATTER MODEL
# ==========================================
print("Running DM model and calculating P...")
n_m3 = n_cm3 * 1e6
gas_mass_kg = mass_gas * MSUN_TO_KG
gas_rho_kg_m3 = gas_rho_msun_kpc3 * MSUN_KPC3_TO_KG_M3

# Volumes and Distances
V_i = gas_mass_kg / gas_rho_kg_m3
d_avg = np.cbrt(1.0 / n_m3)

# Solve for P using alpha = 2 (Standard Model)
alpha_std = 2.0
sum_factor = np.sum(V_i / (c**3 * d_avg**alpha_std))
P = target_dm_mass_kg / sum_factor
print(f"Calculated Power P: {P:.2e} Watts")

# Calculate predicted DM mass per gas particle
dm_mass_kg_pred = P * (V_i / (c**3 * d_avg**alpha_std))
dm_mass_msun_pred = dm_mass_kg_pred / MSUN_TO_KG

# ==========================================
# 5. CALCULATE ROTATION CURVES
# ==========================================
print("Calculating velocity curves...")
# Individual components
_, v_stars = get_component_velocity(r_stars, mass_stars)
_, v_gas = get_component_velocity(r_gas, mass_gas)
r_dm_t, v_dm_true = get_component_velocity(r_dm_true, mass_dm_true)
r_dm_p, v_dm_pred = get_component_velocity(r_gas, dm_mass_msun_pred)

# Total curves (combining components)
all_r_true = np.concatenate([r_stars, r_gas, r_dm_true])
all_m_true = np.concatenate([mass_stars, mass_gas, mass_dm_true])
r_tot_true, v_tot_true = get_component_velocity(all_r_true, all_m_true)

all_r_pred = np.concatenate([r_stars, r_gas, r_gas]) # r_gas used twice (gas + pred DM)
all_m_pred = np.concatenate([mass_stars, mass_gas, dm_mass_msun_pred])
r_tot_pred, v_tot_pred = get_component_velocity(all_r_pred, all_m_pred)

# ==========================================
# 6. PLOTTING THE RESULTS
# ==========================================
print("Generating graphs...")
fig, axes = plt.subplots(3, 1, figsize=(10, 18))

# Graph 1: Total Rotation Curve
axes[0].plot(r_tot_true, v_tot_true, label='True FIRE Simulation', color='black', lw=2)
axes[0].plot(r_tot_pred, v_tot_pred, label='Your Model (Baryons + Pred DM)', color='crimson', ls='--', lw=2)
axes[0].set_title('Total Rotation Curve Comparison', fontsize=14)
axes[0].set_ylabel('Velocity (km/s)', fontsize=12)
axes[0].legend()

# Graph 2: Component Decomposition
axes[1].plot(r_dm_t, v_dm_true, label='True DM Halo', color='black', lw=2)
axes[1].plot(r_dm_p, v_dm_pred, label=f'Predicted DM Halo (α={alpha_std})', color='crimson', ls='--', lw=2)
axes[1].plot(np.sort(r_stars), v_stars, label='Stars', color='goldenrod', ls='-.')
axes[1].plot(np.sort(r_gas), v_gas, label='Gas', color='teal', ls='-.')
axes[1].set_title('Velocity Contributions by Component', fontsize=14)
axes[1].set_ylabel('Velocity Contribution (km/s)', fontsize=12)
axes[1].legend()

# Graph 3: Tuning the Power-Law (Alpha)
axes[2].plot(r_dm_t, v_dm_true, label='True DM Halo', color='black', lw=2)
alpha_test_values = [1.0, 1.5, 2.0, 2.5]
for a in alpha_test_values:
    # Re-calculate P and masses for this alpha
    sf = np.sum(V_i / (c**3 * d_avg**a))
    P_a = target_dm_mass_kg / sf
    dm_m = (P_a * (V_i / (c**3 * d_avg**a))) / MSUN_TO_KG
    r_a, v_a = get_component_velocity(r_gas, dm_m)
    axes[2].plot(r_a, v_a, label=f'Model ($\alpha={a}$)', ls='--')

axes[2].set_title('Tuning the Distance Exponent (α)', fontsize=14)
axes[2].set_xlabel('Galactocentric Radius (kpc)', fontsize=12)
axes[2].set_ylabel('DM Velocity Contribution (km/s)', fontsize=12)
axes[2].legend()

# Standardize formatting across all subplots
for ax in axes:
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 350)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()