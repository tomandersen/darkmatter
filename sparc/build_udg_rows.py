"""
Build SPARC-format rows for two gas-rich ultra-diffuse galaxies that are not in SPARC:
AGC 114905 and AGC 242019. Each galaxy is written several times at different distances (AGC 242019)
or inclinations (AGC 114905), with D or i in the galaxy name, so one run of sparc_dm_bruteCGM.py
shows how the fit changes with these observational nuisance parameters.

Outputs (same column layout as the SPARC files):
  ./sparc/MassModels_UDG_additions_SigmaGas.mrt   -> rows like MassModels_Lelli2016_SigmaGas.mrt
  ./sparc/SPARC_UDG_additions.mrt.txt             -> rows like SPARC_Lelli2016c.mrt.txt

Conventions matched to SPARC / calc_gas_checkit.py:
  - Vgas includes helium with factor 1.33.
  - Vdisk is for M/L = 1 (sparc_dm_bruteCGM.py then applies Upsilon_disk = 0.5).
  - Sigma_gas is HI particles/cm^2, i.e. Sigma_gas(incl. He) / 1.33 converted to protons/cm^2.

Scaling with distance D (factor f = D/D0): R, Rdisk, Reff ~ f; Vgas, Vdisk ~ sqrt(f); L, MHI ~ f^2;
Sigma_gas, surface brightness and Vobs unchanged.
Scaling with inclination i: Vobs, e_Vobs ~ sin(i0)/sin(i); everything else unchanged.

AGC 242019 (Shi et al. 2021, ApJ 909, 20; arXiv:2101.01282, Table 2):
  Tabulated R, Vcirc (pressure-support corrected), Vgas, Vstar at D = 30.8 Mpc, i ~ 72 deg.
  Vgas rescaled from Shi's He factor 1.36 to 1.33. Vstar is for Upsilon_3.6 = 0.6, converted to M/L = 1.
  Sigma_gas is obtained by inverting Vgas with the same regularized thin-disk inversion used for SPARC.
  D = 30.8 is the Hubble-flow distance; 12.5 is Lelli 2024's MOND posterior.

AGC 114905 (Mancera Pina et al. 2024, arXiv:2404.06537):
  Sigma_gas(R) and Sigma_star(R) from the published analytic profiles (Eqs. 9, 10), D = 78.7 Mpc, i = 31 deg.
  Vgas and Vdisk computed from those profiles with the thin-disk gravity matrix in calc_gas_checkit.py.
  Stellar masses are r-band based; Vdisk is stored so that Upsilon_disk = 0.5 reproduces the published M*.
  i = 31 is the published HI-kinematic inclination; ~15 is Lelli 2024's MOND posterior.
  *** Vobs is PROVISIONAL: digitized from Lelli 2024 (A&A 689, L3) Fig. 3, which shows the
      Mancera Pina 2024 curve rescaled to the MOND posterior (D = 67.59 Mpc, i = 14.71 deg).
      Rescaled here back to D = 78.7 Mpc, i = 31 deg. Replace with the authors' tabulated data when available.
"""
import numpy as np
from calc_gas_checkit import build_gravity_matrix, fit_galaxy_surface_density

MSUN_PC2_TO_PROTONS_CM2 = 1.249e20
HE_SPARC = 1.33
UPS_CODE_DISK = 0.5  # Upsilon_disk applied in sparc_dm_bruteCGM.py

DISTANCES_AGC242019 = [30.8, 25.0, 20.0, 15.0, 12.5]     # Mpc
INCLINATIONS_AGC114905 = [31.0, 25.0, 20.0, 17.0, 15.0]  # deg


def arcsec_to_kpc(D_mpc):
    return D_mpc * 1e3 * np.pi / (180.0 * 3600.0)


def disk_velocity(R_out, sigma_func, R_max):
    """Thin-disk circular velocity (km/s) at R_out for Sigma(R) in Msun/pc^2, via a fine annulus grid."""
    R_fine = np.linspace(R_max / 600.0, R_max, 600)
    M = build_gravity_matrix(R_fine)
    V2 = M @ sigma_func(R_fine)
    V2_out = np.interp(R_out, R_fine, V2)
    return np.sign(V2_out) * np.sqrt(np.abs(V2_out))


def mass_within(sigma_func, R_max):
    R = np.linspace(0.0, R_max, 20001)
    return np.trapezoid(2.0 * np.pi * R * sigma_func(R) * 1e6, R)  # Msun (Sigma per pc^2 -> per kpc^2)


def mass_row(name, D, R, Vobs, eV, Vgas, Vdisk, SBdisk, Sigma_HI_cm2):
    return (f"{name:<11s} {D:6.2f} {R:6.2f} {Vobs:6.2f} {eV:5.2f} {Vgas:6.2f} {Vdisk:6.2f} "
            f"{0.0:6.2f} {SBdisk:7.2f} {0.0:8.2f} {Sigma_HI_cm2:10.3e}")


def prop_row(name, T, D, eD, fD, inc, einc, L, eL, Reff, SBeff, Rd, SBd, MHI, RHI, Vflat, eVflat, Q, ref):
    return (f"{name:>11s} {T:2d} {D:6.2f} {eD:5.2f} {fD:2d} {inc:4.1f} {einc:4.1f} {L:7.3f} {eL:7.3f} "
            f"{Reff:5.2f} {SBeff:8.2f} {Rd:5.2f} {SBd:8.2f} {MHI:7.3f} {RHI:5.2f} {Vflat:5.1f} {eVflat:5.1f} "
            f"{Q:3d} {ref:>14s}")


def variant(base, name, D=None, inc=None):
    """Mass-model rows and property row for one galaxy rescaled to distance D and/or inclination inc."""
    D = base['D'] if D is None else D
    inc = base['inc'] if inc is None else inc
    f = D / base['D']
    v = np.sin(np.radians(base['inc'])) / np.sin(np.radians(inc))
    rows = [mass_row(name, D, *x) for x in zip(base['R'] * f, base['Vobs'] * v, base['eV'] * v,
                                               base['Vgas'] * np.sqrt(f), base['Vdisk'] * np.sqrt(f),
                                               base['SBdisk'], base['Sigma_HI_cm2'])]
    L, Rd, Reff = base['L'] * f**2, base['Rd'] * f, base['Reff'] * f
    prop = prop_row(name, 10, D, base['eD'], 1, inc, base['einc'], L, 0.1 * L, Reff,
                    L * 1e9 / (2 * np.pi * (Reff * 1e3) ** 2), Rd, L * 1e9 / (2 * np.pi * (Rd * 1e3) ** 2),
                    base['MHI'] * f**2, 0.0, 0.0, 0.0, base['Q'], base['ref'])
    return rows, prop


def agc242019():
    R = np.array([0.67, 2.02, 3.36, 4.71, 6.06, 7.40, 8.75])
    Vgas_shi = np.array([0.2, 4.1, 8.2, 11.3, 15.0, 19.1, 22.4])
    Vstar_shi = np.array([1.9, 4.5, 8.3, 10.4, 10.6, 10.1, 9.6])

    Vgas = Vgas_shi * np.sqrt(HE_SPARC / 1.36)
    Sigma_gas = fit_galaxy_surface_density(R, Vgas)          # Msun/pc^2 incl. He
    Sigma_star = fit_galaxy_surface_density(R, Vstar_shi)    # Msun/pc^2 at Upsilon 0.6
    print(f"AGC242019: inverted Sigma_gas (Msun/pc^2) = {np.round(Sigma_gas, 2)}")

    Rd = 2.7  # derived: stellar V peaks near 6 kpc ~ 2.2 Rd (not published)
    return dict(D=30.8, eD=1.5, inc=72.0, einc=1.0, R=R,
                Vobs=np.array([10.6, 15.5, 23.5, 30.7, 35.0, 37.5, 40.9]),
                eV=np.array([1.2, 1.2, 1.3, 1.7, 1.5, 1.6, 2.0]),
                Vgas=Vgas, Vdisk=Vstar_shi / np.sqrt(0.6), SBdisk=Sigma_star / 0.6,
                Sigma_HI_cm2=Sigma_gas / HE_SPARC * MSUN_PC2_TO_PROTONS_CM2,
                L=1.37e8 / 0.6 / 1e9, MHI=0.851, Rd=Rd, Reff=1.678 * Rd, Q=2, ref="Shi21")


def agc114905():
    D, inc = 78.7, 31.0
    k = arcsec_to_kpc(D)

    def sigma_gas(R_kpc):  # Eq. 10, incl. He, R in arcsec
        x = R_kpc / k
        return 3.953 * np.exp(-x / 0.676) * (1.0 + x / 375.220) ** 559.231

    def sigma_star(R_kpc):  # Eq. 9, poly-exponential, truncated at the stellar edge (~7.5 kpc)
        x = R_kpc / k
        poly = 1 + 0.5576 * x - 0.1126 * x**2 + 0.0085 * x**3 - 0.0002 * x**4
        s = 3.6703 * np.exp(-x / 3.6155) * poly
        return np.where((R_kpc <= 7.5) & (s > 0), s, 0.0)

    # Vobs digitized from Lelli 2024 Fig. 3 (D = 67.59 Mpc, i = 14.71 deg), rescaled to D = 78.7, i = 31.
    R_fig = np.array([0.95, 2.85, 4.75, 6.65, 8.55])
    V_fig = np.array([21.6, 36.1, 49.7, 54.6, 54.2])
    e_fig = np.array([3.4, 3.6, 3.5, 4.1, 3.5])
    v_scale = np.sin(np.radians(14.71)) / np.sin(np.radians(inc))
    R = R_fig * D / 67.59

    V_star = disk_velocity(R, sigma_star, 30.0)
    M_gas, M_star = mass_within(sigma_gas, 30.0), mass_within(sigma_star, 10.9)
    print(f"AGC114905: integrated M_gas = {M_gas:.3e} (published 1.38e9), M_star = {M_star:.3e} (published 9e7)")

    return dict(D=D, eD=1.5, inc=inc, einc=2.0, R=R, Vobs=V_fig * v_scale, eV=e_fig * v_scale,
                Vgas=disk_velocity(R, sigma_gas, 30.0), Vdisk=V_star / np.sqrt(UPS_CODE_DISK),
                SBdisk=sigma_star(R) / UPS_CODE_DISK,
                Sigma_HI_cm2=sigma_gas(R) / HE_SPARC * MSUN_PC2_TO_PROTONS_CM2,
                L=9e7 / UPS_CODE_DISK / 1e9, MHI=1.04, Rd=2.80 / 1.678, Reff=2.80, Q=3, ref="MP24,Le24")


if __name__ == "__main__":
    a, b = agc242019(), agc114905()
    out = [variant(a, f"AGC242019_D{D:g}", D=D) for D in DISTANCES_AGC242019]
    out += [variant(b, f"AGC114905_i{i:g}", inc=i) for i in INCLINATIONS_AGC114905]

    with open("./sparc/MassModels_UDG_additions_SigmaGas.mrt", "w") as f:
        f.write("# AGC242019: Shi+2021 Table 2, distance variants. AGC114905: Mancera Pina+2024 profiles, "
                "inclination variants; Vobs PROVISIONAL (digitized, Lelli 2024 Fig.3)\n")
        f.write("\n".join(r for rows, _ in out for r in rows) + "\n")
    with open("./sparc/SPARC_UDG_additions.mrt.txt", "w") as f:
        f.write("# Properties for the UDG additions. AGC242019 Rdisk derived from Vstar shape; "
                "AGC114905 L is M*/0.5 (r-band M*).\n")
        f.write("\n".join(p for _, p in out) + "\n")
    print(f"Wrote {len(out)} galaxy variants to sparc/MassModels_UDG_additions_SigmaGas.mrt "
          f"and sparc/SPARC_UDG_additions.mrt.txt")
