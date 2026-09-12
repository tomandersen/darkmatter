import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import gizmo_analysis as gizmo
from datetime import datetime
import os

# ==========================================
# 1. CONFIGURATION & CONSTANTS
# ==========================================
SNAPSHOT_NUM = 600
USE_STARS = False         # Toggle stellar contribution to predicted DM

G = 4.3009e-6             # Gravitational constant: kpc * (km/s)^2 / M_sun
c = 299792458.0           # Speed of light: m/s
MSUN_TO_KG = 1.989e30     # Solar masses to kilograms
MSUN_KPC3_TO_KG_M3 = 6.77e-29 # Mass density conversion
alpha_std = 2.0

def get_component_velocity(radii, masses):
    """Sorts particles by radius and calculates circular velocity."""
    sort_idx = np.argsort(radii)
    r_sorted = radii[sort_idx]
    m_sorted = masses[sort_idx]
    
    enclosed_mass = np.cumsum(m_sorted)
    v_circ = np.sqrt((G * enclosed_mass) / (r_sorted + 1e-6)) # Avoid div by zero
    return r_sorted, v_circ


def process_sim(sim_folder, plot_x_limit=100.0, plot_y_limit=350, power=30.0, density_threshold=0.0002):
    print(f"{sim_folder} Rotation Curves gas generated DM. rho > {density_threshold:.1e}/cm^-3")
    
    # ==========================================
    # 2. LOAD DATA & EXTRACT PROPERTIES
    # ==========================================
    print("Loading snapshot data...")
    part = gizmo.io.Read.read_snapshots(
        ['star', 'gas', 'dark'], 
        'index', 
        SNAPSHOT_NUM, 
        assign_hosts=True, 
        simulation_directory=f"./fire2/galaxies/{sim_folder}"
    ) 
     
    r_stars = part['star'].prop('host.distance.spherical')[:, 0]
    r_gas_all = part['gas'].prop('host.distance.spherical')[:, 0]
    r_dm_all = part['dark'].prop('host.distance.spherical')[:, 0]
    pos_stars_all = part['star'].prop('host.distance') 
    mass_stars = part['star']['mass']
    mass_gas_all = part['gas']['mass']
    mass_dm_all = part['dark']['mass']
    
    # ==========================================
    # 3. FILTER GAS & TRUE DARK MATTER
    # ==========================================
    print("Filtering gas and isolating 100 kpc volume...")
    gas_density_cm3 = part['gas'].prop('number.density') 
    
    # Gas filter mask
    low_density_mask = (gas_density_cm3 > density_threshold)
    r_gas_f = r_gas_all[low_density_mask]
    mass_gas_f = mass_gas_all[low_density_mask]
    n_cm3 = gas_density_cm3[low_density_mask]
    gas_rho_msun_kpc3 = part['gas'].prop('density')[low_density_mask]
    
    # Star filter mask
    r_stars_f = r_stars
    mass_stars_f = mass_stars
    pos_stars_f = pos_stars_all
    
    # True DM filter mask
    r_dm_true = r_dm_all
    mass_dm_true = mass_dm_all
    target_dm_mass_kg = np.sum(mass_dm_true, dtype=np.float64) * MSUN_TO_KG
    
    # ==========================================
    # 4. RUN YOUR DARK MATTER MODEL (GAS + STARS)
    # ==========================================
    print("Running DM model for Gas...")
    n_m3_gas = n_cm3.astype(np.float64) * 1e6
    gas_mass_kg = mass_gas_f.astype(np.float64) * MSUN_TO_KG
    gas_rho_kg_m3 = gas_rho_msun_kpc3.astype(np.float64) * MSUN_KPC3_TO_KG_M3
    
    V_i_gas = gas_mass_kg / gas_rho_kg_m3
    d_avg_gas = np.cbrt(1.0 / n_m3_gas)
    sum_factor_star = 0.0
    
    if USE_STARS:
        print("Calculating local density for Stars using KD-Tree...")
        tree = cKDTree(pos_stars_f)
        distances, _ = tree.query(pos_stars_f, k=17, workers=-1) 
        r_16 = distances[:, 16].astype(np.float64) 
    
        V_sphere_kpc3 = (4.0 / 3.0) * np.pi * (r_16**3)
        star_rho_kg_m3 = ((16.0 * mass_stars_f.mean() * MSUN_TO_KG) / V_sphere_kpc3) * MSUN_KPC3_TO_KG_M3
        V_i_star = (mass_stars_f.astype(np.float64) * MSUN_TO_KG) / star_rho_kg_m3
        mass_baryon_kg = 1.67e-27 
        n_m3_star = star_rho_kg_m3 / mass_baryon_kg
        d_avg_star = np.cbrt(1.0 / n_m3_star)
        sum_factor_star = np.sum(V_i_star / (c**3 * d_avg_star**alpha_std)) if USE_STARS else 0.0
    
    
    print("Calculating global Power P...")
    
    sum_factor_gas = np.sum(V_i_gas / (c**3 * d_avg_gas**alpha_std))
    sum_factor_total = sum_factor_gas + sum_factor_star
    
    calc_P = target_dm_mass_kg / sum_factor_total
    print(f"Calculated Shared Power P: {calc_P:.2e} Watts")
    print(f"Using forced Shared Power P: {power:.2e} Watts")
    
    dm_mass_msun_pred_gas = (power * (V_i_gas / (c**3 * d_avg_gas**alpha_std))) / MSUN_TO_KG
    if USE_STARS:
        dm_mass_msun_pred_star = (power * (V_i_star / (c**3 * d_avg_star**alpha_std))) / MSUN_TO_KG
    
    # ==========================================
    # 5. CALCULATE ROTATION CURVES
    # ==========================================
    print("Calculating velocity curves...")
    _, v_stars = get_component_velocity(r_stars, mass_stars)
    _, v_gas = get_component_velocity(r_gas_all, mass_gas_all)
    r_dm_t, v_dm_true = get_component_velocity(r_dm_true, mass_dm_true)
    r_dm_p, v_dm_pred = get_component_velocity(r_gas_f, dm_mass_msun_pred_gas)
    
    all_r_true = np.concatenate([r_stars, r_gas_all, r_dm_true])
    all_m_true = np.concatenate([mass_stars, mass_gas_all, mass_dm_true])
    r_tot_true, v_tot_true = get_component_velocity(all_r_true, all_m_true)
     
    if USE_STARS:
        all_r_pred = np.concatenate([r_stars_f, r_gas_all, r_stars_f, r_gas_f]) 
        all_m_pred = np.concatenate([mass_stars_f, mass_gas_all, dm_mass_msun_pred_star, dm_mass_msun_pred_gas])
    else:
        all_r_pred = np.concatenate([r_stars_f, r_gas_all, r_gas_f]) 
        all_m_pred = np.concatenate([mass_stars_f, mass_gas_all, dm_mass_msun_pred_gas])
        
    r_tot_pred, v_tot_pred = get_component_velocity(all_r_pred, all_m_pred)
    
    # ==========================================
    # 6. PLOTTING AND SAVING THE RESULTS
    # ==========================================
    print("Generating and saving graphs...")
    S = 100 
    
    settings_string = f"gas-{density_threshold:.1e}_P-{power:.1e}W"
    component_filename = f"{sim_folder}-{settings_string}-components.png"
    component_path = f"./fire2/components/{settings_string}/{component_filename}"
    os.makedirs(os.path.dirname(component_path), exist_ok=True)
    
    curve_filename = f"{sim_folder}-{settings_string}-curve.png"
    curve_path = f"./fire2/curves/{settings_string}/{curve_filename}"
    os.makedirs(os.path.dirname(curve_path), exist_ok=True)
    
    # ------------------------------------------
    # Graph 1: Total Rotation Curve
    # ------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(r_tot_true[::S], v_tot_true[::S], label='Full FIRE Simulation', color='black', lw=2)
    plt.plot(r_tot_pred[::S], v_tot_pred[::S], label=f"Baryons + DM Prediction (P = {power:.1f} Watts)", color='crimson', ls='--', lw=2)
     
    plt.title(f"{sim_folder} Rotation Curves gas generated DM. rho > {density_threshold:.1e}/cm^-3", fontsize=14)
    plt.xlabel('Galactocentric Radius (kpc)', fontsize=12)
    plt.ylabel('Velocity (km/s)', fontsize=12) 
    plt.xlim(0, plot_x_limit)
    plt.ylim(0, plot_y_limit)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    plt.figtext(0.99, 0.01, f"Generated: {current_time}", ha="right", va="bottom", fontsize=9, color="gray", alpha=0.7)
    plt.savefig(curve_path, dpi=300)
    plt.close() # Closes the figure to free up your memory!
    print(f"Saved {curve_path}")
    
    # ------------------------------------------
    # Graph 2: Component Decomposition
    # ------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(r_dm_t[::S], v_dm_true[::S], label='FIRE DM Halo', color='black', lw=2)
    plt.plot(r_dm_p[::S], v_dm_pred[::S], label=f"Predicted DM Halo (P = {power:.1f} Watts)", color='crimson', ls='--', lw=2)
    plt.plot(np.sort(r_stars)[::S], v_stars[::S], label='Stars', color='goldenrod', ls='-.')
    plt.plot(np.sort(r_gas_all)[::S], v_gas[::S], label='Gas', color='teal', ls='-.')
    
    plt.title(f"{sim_folder} Component Curves, rho > {density_threshold:.1e}/cm^-3", fontsize=14)
    plt.xlabel('Galactocentric Radius (kpc)', fontsize=12)
    plt.ylabel('Velocity Contribution (km/s)', fontsize=12)
    plt.xlim(0, plot_x_limit)
    plt.ylim(0, plot_y_limit)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    plt.figtext(0.99, 0.01, f"Generated: {current_time}", ha="right", va="bottom", fontsize=9, color="gray", alpha=0.7)
    plt.savefig(component_path, dpi=300)
    plt.close()
    print(f"Saved {component_path}")


if __name__ == "__main__":
    #sim_folders = [ "m11i_res7100", "m12i_res7100"]
    
    sim_folders = [
        "m11i_res7100", "m12i_res7100", "m10q_res30", "m11h_res7100", 
        "m11e_res7100", "m12m_res7100", "m09_res30", "m12r_res7100", 
        "m12c_res7100", "m11b_res2100", "m11q_res880", "m12w_res7100"
    ]


    for folder in sim_folders:
        process_sim(
            sim_folder=folder, 
            plot_x_limit=100.0, 
            plot_y_limit=350, 
            power=30.0, 
            density_threshold=0.0002
        )