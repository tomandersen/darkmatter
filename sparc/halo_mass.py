import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

def plot_converged_halo_profiles():
    # ---------------------------------------------------------
    # 1. Setup X-axis (Radius in kpc)
    # ---------------------------------------------------------
    r = np.logspace(-2, 2.5, 1000)

    # ---------------------------------------------------------
    # 2. Unified Physical Parameters (Milky Way-sized)
    # ---------------------------------------------------------
    rho_s_nfw = 0.35      # m_p/cm^3
    rs_nfw = 18.0         # kpc
    rho_0_iso = 2.5       # m_p/cm^3
    rc_dm = 3.0           # kpc
    n0_gas = 5e-3         # particles/cm^3
    rc_gas = 3.5          # kpc
    beta = 0.5            
    P_c3_factor = 10.0    
    density_cut = 1e-5    # Gas density cutoff (n_c)

    # ---------------------------------------------------------
    # 3. Dynamic r_200 Calculation
    # ---------------------------------------------------------
    H0_km_s_Mpc = 70.0
    G_cgs = 6.6743e-8
    Mpc_to_cm = 3.086e24
    mp_to_g = 1.672e-24
    
    H0_cgs = (H0_km_s_Mpc * 1e5) / Mpc_to_cm 
    rho_crit_mp_cm3 = ((3 * H0_cgs**2) / (8 * np.pi * G_cgs)) / mp_to_g
    target_density = 200 * rho_crit_mp_cm3

    mass_nfw_raw = 4 * np.pi * rho_s_nfw * (rs_nfw**3) * \
                   (np.log(1 + r/rs_nfw) - (r/rs_nfw)/(1 + r/rs_nfw))
    avg_rho_nfw = mass_nfw_raw / ((4/3) * np.pi * r**3)
    r_200 = np.interp(target_density, avg_rho_nfw[::-1], r[::-1])

    # ---------------------------------------------------------
    # 4. Density Profiles with STEEPER Taper (Converging Mass)
    # ---------------------------------------------------------
    rho_nfw = rho_s_nfw / ((r / rs_nfw) * (1 + r / rs_nfw)**2)
    rho_iso = rho_0_iso / (1 + (r / rc_dm)**2)
    rho_gas = n0_gas * (1 + (r / rc_gas)**2)**(-1.5 * beta)
    
    # Steeper taper to guarantee mass convergence (k > 1.33)
    k = 2.0 
    taper = 1.0 / (1.0 + (density_cut / rho_gas)**k)
    
    rho_custom = P_c3_factor * (rho_gas)**(2/3) * taper

    # ---------------------------------------------------------
    # 5. Mass Integration in Solar Masses
    # ---------------------------------------------------------
    kpc_to_cm = 3.086e21
    Msun_to_g = 1.989e33
    conversion_factor = (mp_to_g * (kpc_to_cm)**3) / Msun_to_g

    mass_nfw = mass_nfw_raw * conversion_factor
    mass_iso = 4 * np.pi * rho_0_iso * (rc_dm**3) * (r/rc_dm - np.arctan(r/rc_dm)) * conversion_factor

    integrand_gas = 4 * np.pi * (r**2) * rho_gas * conversion_factor
    integrand_custom = 4 * np.pi * (r**2) * rho_custom * conversion_factor
    
    mass_gas = np.insert(cumulative_trapezoid(integrand_gas, r), 0, 0)
    mass_custom = np.insert(cumulative_trapezoid(integrand_custom, r), 0, 0)

    # ---------------------------------------------------------
    # 6. Plotting
    # ---------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    r_cut_idx = np.where(rho_gas < density_cut)[0]
    r_cut = r[r_cut_idx[0]] if len(r_cut_idx) > 0 else None

    # Density
    ax1.plot(r, rho_nfw, label='1) NFW DM', color='black', linestyle='-')
    ax1.plot(r, rho_iso, label='2) Isothermal DM', color='blue', linestyle='--')
    ax1.plot(r, rho_gas, label='3) CGM Gas', color='green', linestyle='-.')
    ax1.plot(r, rho_custom, label=f'4) Custom DM (Taper k={k})', color='red', linestyle='-', linewidth=2.5)

    if r_cut: ax1.axvline(r_cut, color='gray', linestyle=':', label=f'Taper Midpoint ($n=n_c$)')
    ax1.axvline(r_200, color='purple', linestyle='-', alpha=0.5, label=f'Dynamic $r_{{200}}$ ({r_200:.0f} kpc)')

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim(0.1, 300)
    ax1.set_ylim(1e-6, 1e2)
    ax1.set_xlabel('Radius $r$ (kpc)')
    ax1.set_ylabel('Density $\\rho$ ($m_p$ / cm$^3$)')
    ax1.set_title('Density Profiles')
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend()

    # Enclosed Mass
    ax2.plot(r, mass_nfw, label='1) NFW Mass', color='black', linestyle='-')
    ax2.plot(r, mass_iso, label='2) Isothermal DM Mass', color='blue', linestyle='--')
    ax2.plot(r, mass_gas, label='3) CGM Gas Mass', color='green', linestyle='-.')
    ax2.plot(r, mass_custom, label=f'4) Custom DM Mass (Taper k={k})', color='red', linestyle='-', linewidth=2.5)

    if r_cut: ax2.axvline(r_cut, color='gray', linestyle=':', label=f'Taper Midpoint ($n=n_c$)')
    ax2.axvline(r_200, color='purple', linestyle='-', alpha=0.5, label=f'Dynamic $r_{{200}}$ ({r_200:.0f} kpc)')

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlim(0.1, 300)
    ax2.set_ylim(1e7, 5e12)
    ax2.set_xlabel('Radius $r$ (kpc)')
    ax2.set_ylabel('Enclosed Mass $M(<r)$ ($M_\\odot$)')
    ax2.set_title('Enclosed Mass Profiles (Converged)')
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_converged_halo_profiles()