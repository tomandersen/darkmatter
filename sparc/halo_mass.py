import numpy as np
import matplotlib.pyplot as plt

def plot_physical_halo_profiles():
    # ---------------------------------------------------------
    # 1. Setup X-axis (Radius in kpc, from 0.1 to ~316 kpc)
    # ---------------------------------------------------------
    r = np.logspace(-1, 2.5, 500) 

    # ---------------------------------------------------------
    # 2. Physically Realistic Parameters (Milky Way-sized)
    # ---------------------------------------------------------
    # NFW DM Parameters
    rho_s_nfw = 0.35      # Scale density (m_p/cm^3)
    rs_nfw = 18.0         # Scale radius (kpc)
    
    # Isothermal DM Parameters
    rho_0_iso = 2.5       # Central DM density (m_p/cm^3)
    rc_dm = 3.0           # DM core radius (kpc)
    
    # CGM Gas Parameters (Beta model)
    n0_gas = 5e-3         # Central gas density (particles/cm^3)
    rc_gas = 3.5          # Gas core radius (kpc)
    beta = 0.5            # Beta parameter
    
    # Custom DM Parameters
    # P_c3_factor set to ~85 so the custom DM core matches the Isothermal DM core
    P_c3_factor = 10.0     
    density_cut = 1e-5    # Gas density cutoff where DM generation stops

    # ---------------------------------------------------------
    # 3. Calculate Density Profiles
    # ---------------------------------------------------------
    # Profile 1: NFW Dark Matter
    rho_nfw = rho_s_nfw / ((r / rs_nfw) * (1 + r / rs_nfw)**2)

    # Profile 2: Isothermal Dark Matter
    rho_iso = rho_0_iso / (1 + (r / rc_dm)**2)
    
    # Profile 3: CGM Gas (Beta-model)
    rho_gas = n0_gas * (1 + (r / rc_gas)**2)**(-1.5 * beta)

    # Profile 4: Custom DM Model (with density cut)
    rho_custom = P_c3_factor * (rho_gas)**(2/3)
    rho_custom[rho_gas < density_cut] = 1e-10  # Truncate where gas is too rarefied

    # ---------------------------------------------------------
    # 4. Plotting
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 7))

    plt.plot(r, rho_nfw, label='1) NFW Dark Matter', color='black', linestyle='-')
    plt.plot(r, rho_iso, label='2) Isothermal Dark Matter', color='blue', linestyle='--')
    plt.plot(r, rho_gas, label='3) CGM Gas (Beta-model)', color='green', linestyle='-.')
    plt.plot(r, rho_custom, label='4) Custom DM Model', color='red', linestyle='-', linewidth=2.5)

    # Mark the cutoff point and the Solar position
    cut_idx = np.where(rho_gas < density_cut)[0]
    if len(cut_idx) > 0:
        r_cut = r[cut_idx[0]]
        plt.axvline(r_cut, color='gray', linestyle=':', label=f'Density Cut ($r \\approx {r_cut:.1f}$ kpc)')

    plt.axvline(8.0, color='orange', linestyle=':', alpha=0.7, label='Solar Position (8 kpc)')

    # Formatting
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(1e-6, 1e2)  # Expanded to show the DM dominance
    plt.xlim(0.1, 300)
    
    plt.xlabel('Radius $r$ (kpc)', fontsize=12)
    plt.ylabel('Density $\\rho$ ($m_p$ / cm$^3$)', fontsize=12)
    plt.title('Physically Accurate Density Profiles (Milky Way Mass)', fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=11)
    
    plt.show()

if __name__ == "__main__":
    plot_physical_halo_profiles()