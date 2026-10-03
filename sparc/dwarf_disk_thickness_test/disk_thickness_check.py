from itertools import count
import numpy as np
import os
from scipy.integrate import quad
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# --- Configuration ---
TARGET_GALAXIES = ["DDO154", "NGC3741"]
TEST_POWER_W = 8.7              # User preferred universal power
DWARF_THICKNESS_KPC = 1.0       # User preferred puffiness

Upsilon_disk = 0.5
sqrt_Upsilon_disk = np.sqrt(Upsilon_disk)

# 1. Setup and Parse the Data
kinematics_file = "./sparc/MassModels_Lelli2016c.mrt"
output_dir = "./sparc/dwarf_exponential_test"
os.makedirs(output_dir, exist_ok=True)

galaxies = {}

with open(kinematics_file, 'r') as f:
    for line in f:
        parts = line.split()
        if len(parts) == 10 and not line.startswith('---') and not line.startswith('Byte'):
            try:
                gal_name = parts[0]
                if gal_name not in TARGET_GALAXIES:
                    continue
                    
                R = float(parts[2])
                Vobs = float(parts[3])
                eVobs = float(parts[4]) if float(parts[4]) > 0 else 1.0
                Vgas = float(parts[5])
                Vdisk = float(parts[6]) * sqrt_Upsilon_disk
                Vbul = float(parts[7]) 
                
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

# 3. Process and Plot
count = 0
for DWARF_THICKNESS_KPC in [0.01,0.04, 0.1, 0.2, 0.5,0.7,0.85,1.0,1.5,2.0,2.5,3.0,4.0,5.0, 10, 20]:   
    for gal in TARGET_GALAXIES:
        if gal not in galaxies:
            continue
            
        print(f"Processing {gal} with Exponential Profile (hz = {DWARF_THICKNESS_KPC} kpc)...")
        g = galaxies[gal]
        R_m = g['R'] * kpc_to_m
        N = len(R_m)
        
        V_gas_ms = g['Vgas'] * 1000.0
        V_disk_ms = g['Vdisk'] * 1000.0
        V_bul_ms = g['Vbul'] * 1000.0
        hz_m = DWARF_THICKNESS_KPC * kpc_to_m
        
        w = np.zeros(N)
        if N > 1:
            w[0] = (R_m[1] - R_m[0]) / 2.0
            w[-1] = (R_m[-1] - R_m[-2]) / 2.0
            for j in range(1, N-1):
                w[j] = (R_m[j+1] - R_m[j-1]) / 2.0
        else:
            w[0] = 1.0

        M_geom = np.zeros((N, N))
        for i_idx in range(N):
            for j_idx in range(N):
                def theta_integrand(theta):
                    num = R_m[i_idx] - R_m[j_idx] * np.cos(theta)
                    # Geometric distance kernel for radial force
                    den = (R_m[i_idx]**2 + R_m[j_idx]**2 - 2 * R_m[i_idx] * R_m[j_idx] * np.cos(theta) + hz_m**2)**1.5
                    return num / den
                ang_int, _ = quad(theta_integrand, 0, 2 * np.pi)
                M_geom[i_idx, j_idx] = G * R_m[i_idx] * R_m[j_idx] * ang_int * w[j_idx]
                
        valid = R_m > 0
        Sigma_b = np.zeros_like(R_m)
        V_bar_sq = np.clip(V_disk_ms**2 + V_bul_ms**2 + np.sign(V_gas_ms)*(V_gas_ms**2), 0, None)
        Sigma_b[valid] = V_bar_sq[valid] / (2 * np.pi * G * R_m[valid])
        
        # Calculate midplane baryon density (rho_0) based on exponential profile
        rho_0 = Sigma_b / (2 * hz_m)
        n_0 = rho_0 / m_p 
        
        # NEW: Analytically integrated Dark Mass surface density for exponential profile
        # The integral of e^(-2/3 * z/hz) yields a 1.5x multiplier over the uniform slab
        rho_dm_0 = (TEST_POWER_W / c**3) * (n_0 ** (2/3))
        Sigma_dm_exp = 3.0 * hz_m * rho_dm_0
        
        V_dm_sq_ms = M_geom @ Sigma_dm_exp
        V_dm_sq_ms = np.clip(V_dm_sq_ms, 0, None)
        
        V_dm_kms = np.sqrt(V_dm_sq_ms) / 1000.0
        V_tot_kms = np.sqrt(np.clip((V_bar_sq + V_dm_sq_ms), 0, None)) / 1000.0
        
        plt.figure(figsize=(8, 5))
        plt.plot(g['R'], g['Vgas'], 'b:', label='Gas')
        plt.plot(g['R'], g['Vdisk'], 'y:', label='Stars')
        plt.plot(g['R'], V_dm_kms, 'm--', label=f'Dark Mass (Exp. Profile, P={TEST_POWER_W}W)')
        plt.plot(g['R'], V_tot_kms, 'r-', linewidth=2, label='Total DM Predicted')
        plt.errorbar(g['R'], g['Vobs'], yerr=g['eVobs'], fmt='ko', label='Observed Data', capsize=2)
        
        plt.xlabel('Radius (kpc)')
        plt.ylabel('Velocity (km/s)')
        plt.title(f'{gal} Geometry Test (Exponential Profile, hz = {DWARF_THICKNESS_KPC} kpc)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"{gal}_{count}thick_disk_test-{DWARF_THICKNESS_KPC}kpc.png"), bbox_inches='tight', dpi=300)
        plt.close()
        count = count + 1

print(f"Done. Check /{output_dir} for the results.")