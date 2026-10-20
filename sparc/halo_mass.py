import numpy as np
import matplotlib.pyplot as plt

def plot_halo_profiles():
    # ---------------------------------------------------------
    # 1. Setup X-axis (Radius in kpc)
    # ---------------------------------------------------------
    r = np.logspace(-1, 2.5, 500) # From 0.1 kpc to ~316 kpc

    # ---------------------------------------------------------
    # 2. Parameters (Adjustable to fit specific galaxy models)
    # ---------------------------------------------------------
    # CGM Gas Parameters (Beta model)
    n0_gas = 1e-2         # Central gas density (particles/cm^3)
    rc_gas = 10.0         # Gas core radius (kpc)
    beta = 0.6            # Beta parameter
    
    # NFW DM Parameters
    rho_s_nfw = 5e-3      # Scale density (m_p/cm^3 equivalent)
    rs_nfw = 15.0         # Scale radius (kpc)
    
    # Isothermal DM Parameters (Pseudo-isothermal sphere)
    rho_0_iso = 5e-2      # Central DM density (m_p/cm^3)
    rc_dm = 3.0           # DM core radius (kpc) - often smaller than gas core
    
    # Custom DM Parameters
    # P/c^3 scaling factor chosen to align visually with other DM profiles
    P_c3_factor = 1.0     
    density_cut = 3e-5    # Gas density cutoff (particles/cm^3)

    # ---------------------------------------------------------
    # 3. Calculate Density Profiles
    # ---------------------------------------------------------
    # Profile 1: CGM Gas (Beta model)
    rho_gas = n0_gas * (1 + (r / rc_gas)**2)**(-1.5 * beta)

    # Profile 2: NFW Dark Matter
    rho_nfw = rho_s_nfw / ((r / rs_nfw) * (1 + r / rs_nfw)**2)

    # Profile 3: Isothermal Dark Matter
    rho_iso = rho_0_iso / (1 + (r / rc_dm)**2)

    # Profile 4: Custom DM Model (with density cut)
    # Start with the base formula: (P/c^3) * n^(2/3)
    rho_custom = P_c3_factor * (rho_gas)**(2/3)
    
    # Apply the physical density cut: where gas is too rarefied, DM is 0
    # We use a very small number (1e-10) instead of 0 for log-log plotting
    rho_custom[rho_gas < density_cut] = 1e-10

    # ---------------------------------------------------------
    # 4. Plotting
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 7))

    plt.plot(r, rho_nfw, label='1) NFW Dark Matter', color='black', linestyle='-')
    plt.plot(r, rho_iso, label='2) Isothermal Dark Matter', color='blue', linestyle='--')
    plt.plot(r, rho_gas, label='3) CGM Gas (Beta-model)', color='green', linestyle='-.')
    plt.plot(r, rho_custom, label='4) Custom DM Model', color='red', linestyle='-', linewidth=2.5)

    # Mark the cutoff point
    cut_idx = np.where(rho_gas < density_cut)[0]
    if len(cut_idx) > 0:
        r_cut = r[cut_idx[0]]
        plt.axvline(r_cut, color='gray', linestyle=':', label=f'Density Cut ($r \\approx {r_cut:.1f}$ kpc)')

    # Formatting
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(1e-6, 1e0) # Focused on relevant densities
    plt.xlim(0.1, 300)
    
    plt.xlabel('Radius $r$ (kpc)', fontsize=12)
    plt.ylabel('Density $\\rho$ ($m_p$ / cm$^3$)', fontsize=12)
    plt.title('Comparison of Density Profiles', fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=11)
    
    plt.show()

if __name__ == "__main__":
    plot_halo_profiles()