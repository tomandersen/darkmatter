# Single-galaxy "3D gas test" for the dark mass model.
#
# Question: if we feed the brute-force model the gas we actually measure (HI from THINGS, H2 from HERACLES CO,
# plus a stated warm ionized layer), each phase with its own vertical profile, does the rotation curve get better,
# and how much of the invented spherical core gas is still needed?
#
# Variants (all at fixed P = 9.2 W, CGM off, same grid):
#   a0: SPARC Sigma_gas column (from Vgas), no core gas
#   a : SPARC Sigma_gas column + core gas (0.6 / beta 1.93 / r_core_factor 2.5)   <- the current canonical model
#   b : measured HI + H2 + ionized, no core gas
#   c : measured HI + H2 + ionized + core gas
# plus sensitivity runs (alpha_CO x0.5 and x2, no ionized gas) and a scan of the core gas central density.
#
# Needs sparc/gas3d/<GAL>_profiles.csv from gas3d_build_profiles.py. Run from the repo root:
#   .venv/bin/python sparc/gas3d_test.py
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sparc_dm_bruteCGM import run_sparc_dm_brute_cgm, core_gas_density_correction

GALAXIES = ["NGC6946", "NGC2403"]
POWER = 9.2
CORE = dict(add_core_gas_central_density=0.60, add_core_gas_beta=1.93, r_core_factor_gas=2.5)
NO_CGM = dict(add_real_cgm_central_density=0.0, r_core_factor_cgm=0.20, cgm_beta=0.50)
NUM_MINUTES = 8
Z_STRETCH_KPC = 0.02   # sinh-spaced z layers: ~2 pc thick at the mid-plane, ~1 kpc at |z| = 10 kpc
OUT_DIR = "./sparc/gas3d"
RUN_DIR = os.path.join(OUT_DIR, "runs")

kpc_to_m = 3.086e19

# Vertical structure per phase (exponential, n ~ exp(-|z|/h)). Assumptions, not fits:
#   HI:  h = 0.15 + 0.025 R kpc  (~150 pc in the centre flaring to ~650 pc at 20 kpc; THINGS galaxies flare like this, e.g. Bacchini et al. 2019)
#   H2:  h = 0.05 kpc            (Milky Way molecular layer, exponential scale height ~50 pc)
#   ionized (warm ionized medium / Reynolds layer): Sigma = F_ION * Sigma_HI, h = 1.0 kpc
#        F_ION = 0.25 from the Milky Way: pulsar dispersion measures give N_e ~ 1.4e20 cm^-2 through the disk vs N_HI ~ 6e20 at the Sun
def h_HI_kpc(R_kpc):
    return 0.15 + 0.025 * R_kpc

H_H2_KPC = 0.05
H_ION_KPC = 1.0
F_ION = 0.25


def load_profiles(gal):
    d = np.genfromtxt(os.path.join(OUT_DIR, f"{gal}_profiles.csv"), delimiter=",", skip_header=2)
    return dict(R=d[:, 0], HI=d[:, 1], H2=d[:, 3], clump_tot=d[:, 7])


def make_density_fn(prof, alpha_scale=1.0, f_ion=F_ION, phases=("HI", "H2", "ion")):
    """Returns fn(R_m, Z_m, Theta_m) -> gas particle density [m^-3] for run_sparc_dm_brute_cgm(supplied_gas_density=...)."""
    def fn(R_m, Z_m, Theta_m):
        R_kpc = R_m[:, 0, 0] / kpc_to_m
        z_kpc = Z_m[0, :, 0] / kpc_to_m
        dz = np.gradient(z_kpc)
        # face-on columns in particles/m^2 (hydrogen only, like the SPARC Sigma_gas column); held flat past the last ring
        S_HI = np.interp(R_kpc, prof["R"], prof["HI"]) * 1e4
        S_H2 = alpha_scale * np.interp(R_kpc, prof["R"], prof["H2"]) * 1e4
        S_ion = f_ion * S_HI
        n = np.zeros(R_m.shape[:2])
        for phase, S, h in (("HI", S_HI, h_HI_kpc(R_kpc)), ("H2", S_H2, np.full_like(R_kpc, H_H2_KPC)), ("ion", S_ion, np.full_like(R_kpc, H_ION_KPC))):
            if phase not in phases:
                continue
            w = np.exp(-np.abs(z_kpc)[None, :] / h[:, None])
            # normalise on the actual (non-uniform) z layers so each phase keeps exactly its measured column
            w /= np.sum(w * dz[None, :], axis=1)[:, None] * kpc_to_m
            n += S[:, None] * w
        return n[:, :, None] * np.ones_like(Theta_m)
    return fn


def run_variant(name, use_measured, core_density, alpha_scale=1.0, f_ion=F_ION, graphs=False, phases=("HI", "H2", "ion")):
    supplied = None
    if use_measured:
        supplied = {gal: make_density_fn(load_profiles(gal), alpha_scale, f_ion, phases) for gal in GALAXIES}
    core = dict(CORE, add_core_gas_central_density=core_density)
    _, _, gals = run_sparc_dm_brute_cgm(**core, **NO_CGM, num_minutes=NUM_MINUTES, output_graphs=graphs, power_override=POWER,
                                        supplied_gas_density=supplied, only_galaxies=GALAXIES,
                                        output_dir_override=os.path.join(RUN_DIR, name), return_galaxies=True,
                                        z_stretch_kpc=Z_STRETCH_KPC)
    return gals


def curves_at_obs(g):
    """Model total, dark mass and gas velocities at the observed radii (model's own 3D gas gravity in every variant)."""
    V_dm_sq = POWER * g['V_dm_sq_base_kms']
    V_bar_sq = np.clip(g['Vdisk']**2 + g['Vbul']**2 + g['Vgas_test'] * np.abs(g['Vgas_test']), 0, None)
    return np.sqrt(V_bar_sq + V_dm_sq), np.sqrt(V_dm_sq), g['Vgas_test']


def mond_from(g, Vgas):
    a0 = 1.2e-10
    R_m = g['R'] * kpc_to_m
    V_bar_sq = np.clip(g['Vdisk']**2 + g['Vbul']**2 + Vgas * np.abs(Vgas), 0, None) * 1e6
    g_N = V_bar_sq / R_m
    return np.sqrt((g_N + np.sqrt(g_N**2 + 4 * g_N * a0)) / 2.0 * R_m) / 1000.0


def residuals(g, V_pred):
    d = g['Vobs'] - V_pred
    return dict(mae=float(np.mean(np.abs(d))), chi2_per_pt=float(np.mean((d / g['eVobs'])**2)), mean_bias=float(np.mean(-d)))


def gas_mass_inside_rmax(g):
    m_p, M_sun = 1.67262e-27, 1.98847e30
    return float(g['sum_all_gas_particles'] * m_p * 1.33 / M_sun)


if __name__ == "__main__":
    os.makedirs(RUN_DIR, exist_ok=True)
    core0 = CORE['add_core_gas_central_density']

    runs = {
        "a0_sparc_nocore":     run_variant("a0_sparc_nocore", False, 0.0),
        "a_sparc_core":        run_variant("a_sparc_core", False, core0, graphs=True),
        "b_measured_nocore":   run_variant("b_measured_nocore", True, 0.0, graphs=True),
        "c_measured_core":     run_variant("c_measured_core", True, core0, graphs=True),
        "b_alpha_x0.5":        run_variant("b_alpha_x0.5", True, 0.0, alpha_scale=0.5),
        "b_alpha_x2":          run_variant("b_alpha_x2", True, 0.0, alpha_scale=2.0),
        "b_no_ionized":        run_variant("b_no_ionized", True, 0.0, f_ion=0.0),
    }

    # Newtonian gas velocity of each measured phase on its own (no core), so the H2 and ionized contributions can be seen in the plots.
    # (The summed gas line adds in quadrature, so a 15 km/s H2 curve under a 35 km/s HI curve barely shows.)
    phase_runs = {ph: run_variant(f"phase_{ph}", True, 0.0, phases=(ph,)) for ph in ("HI", "H2", "ion")}
    core_runs = run_variant("phase_core_only", True, core0, phases=())

    # core gas scan: how much core gas does each gas model want? (n0 = add_core + correction(M_b), so start above the correction)
    scan_values = [0.15, 0.3, 0.45, 0.6, 0.8, 1.0, 1.3]
    scan = {"sparc": {}, "measured": {}}
    for v in scan_values:
        scan["sparc"][v] = run_variant(f"scan_sparc_{v}", False, v)
        scan["measured"][v] = run_variant(f"scan_measured_{v}", True, v)

    results = {}
    for gal in GALAXIES:
        res = {}
        ref = runs["a_sparc_core"][gal]
        for name, gals in runs.items():
            g = gals[gal]
            V_tot, _, _ = curves_at_obs(g)
            res[name] = residuals(g, V_tot)
            res[name]["gas_mass_inside_rmax_Msun"] = gas_mass_inside_rmax(g)
        res["MOND_sparc_gas"] = residuals(ref, ref['V_mond_kms'])
        g_b = runs["b_measured_nocore"][gal]
        res["MOND_measured_gas"] = residuals(g_b, mond_from(g_b, g_b['Vgas_test']))

        res["core_scan"] = {}
        for kind in ("sparc", "measured"):
            rows = []
            for v in scan_values:
                g = scan[kind][v][gal]
                V_tot, _, _ = curves_at_obs(g)
                r = residuals(g, V_tot)
                rows.append(dict(add_core=v, n0_cm3=v + float(core_gas_density_correction(g['gas_plus_tl'])),
                                 mae=r['mae'], gas_mass_inside_rmax_Msun=gas_mass_inside_rmax(g)))
            nocore = res["a0_sparc_nocore" if kind == "sparc" else "b_measured_nocore"]
            rows.insert(0, dict(add_core=0.0, n0_cm3=0.0, mae=nocore['mae'], gas_mass_inside_rmax_Msun=nocore['gas_mass_inside_rmax_Msun']))
            res["core_scan"][kind] = rows
        results[gal] = res

        # ---- plots: the three requested variants side by side ----
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
        for ax, (name, label) in zip(axes, [("a_sparc_core", "(a) SPARC gas + core gas (current)"),
                                            ("b_measured_nocore", "(b) measured 3D gas, no core"),
                                            ("c_measured_core", "(c) measured 3D gas + core gas")]):
            g = runs[name][gal]
            V_dm_grid = np.sqrt(POWER * g['V_dm_sq_base_grid_kms'])
            V_bar_sq_grid = np.clip(g['Vdisk_grid']**2 + g['Vbul_grid']**2 + g['Vgas_test_grid_kms'] * np.abs(g['Vgas_test_grid_kms']), 0, None)
            ax.plot(g['R_grid_kpc'], np.sqrt(V_bar_sq_grid + V_dm_grid**2), 'r-', linewidth=2, label='baryons + dm')
            ax.plot(g['R_grid_kpc'], V_dm_grid, 'm--', linewidth=0.8, label='dark mass')
            ax.plot(g['R_grid_kpc'], g['Vgas_test_grid_kms'], 'g', linewidth=0.8, label='gas (as used, 3D)')
            ax.plot(g['R'], g['Vgas'], 'g', linestyle='dotted', linewidth=0.7, label='gas (Lelli)')
            if name != "a_sparc_core":
                for ph, col in (("HI", 'darkgreen'), ("H2", 'blue'), ("ion", 'orange')):
                    gp = phase_runs[ph][gal]
                    ax.plot(gp['R_grid_kpc'], gp['Vgas_test_grid_kms'], color=col, linestyle='--', linewidth=0.8, label=f'{ph} only (measured)')
            if name != "b_measured_nocore":
                gp = core_runs[gal]
                ax.plot(gp['R_grid_kpc'], gp['Vgas_test_grid_kms'], color='brown', linestyle=':', linewidth=1.0, label='core gas only')
            ax.plot(g['R'], g['Vdisk'], 'violet', linestyle='dashdot', linewidth=0.7, label='disk (Lelli)')
            if np.any(g['Vbul'] > 0):
                ax.plot(g['R'], g['Vbul'], 'red', linestyle='dashed', linewidth=0.7, label='bulge (Lelli)')
            ax.plot(g['R'], g['V_mond_kms'], 'c-.', linewidth=1, label='MOND (SPARC gas)')
            ax.errorbar(g['R'], g['Vobs'], yerr=g['eVobs'], fmt='ko', capsize=1.5, markersize=2, elinewidth=0.6, label='observed')
            r = res[name]
            ax.set_title(f"{gal} {label}\nMAE {r['mae']:.1f} km/s (MOND {res['MOND_sparc_gas']['mae']:.1f})", fontsize=10)
            ax.set_xlabel('Radius (kpc)')
            ax.set_xlim(0, max(g['R']) * 1.02)
            ax.grid(True, linewidth=0.5, alpha=0.7)
        axes[0].set_ylabel('Velocity (km/s)')
        for ax in axes:
            ax.legend(fontsize='x-small', frameon=False)
        plt.savefig(os.path.join(OUT_DIR, f"{gal}_3dgas_variants.png"), bbox_inches='tight', dpi=150)
        plt.close()

        # ---- plot: core gas scan ----
        plt.figure(figsize=(7, 4.5))
        for kind, col in (("sparc", 'k'), ("measured", 'r')):
            rows = res["core_scan"][kind]
            plt.plot([r_['n0_cm3'] for r_ in rows], [r_['mae'] for r_ in rows], col + 'o-', label=f"{kind} gas")
        plt.axhline(res['MOND_sparc_gas']['mae'], color='c', linestyle='-.', label='MOND (SPARC gas)')
        plt.axvline(core0 + float(core_gas_density_correction(ref['gas_plus_tl'])), color='grey', linestyle=':', label='canonical core n0')
        plt.xlabel('core gas central density n0 (cm$^{-3}$)')
        plt.ylabel('mean |Vobs - Vmodel| at observed radii (km/s)')
        plt.title(f'{gal}: how much core gas does each gas model want? (P = {POWER} W)')
        plt.grid(True, linewidth=0.5, alpha=0.7)
        plt.legend(fontsize='small')
        plt.savefig(os.path.join(OUT_DIR, f"{gal}_core_scan.png"), bbox_inches='tight', dpi=150)
        plt.close()

    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    print(json.dumps(results, indent=1))
