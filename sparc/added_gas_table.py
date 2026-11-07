"""Added gas budget per galaxy: integrate the core gas and CGM density profiles.

Reads core_densities.txt and the two Lelli 2016 files, writes a table and graphs to OUTPUT_DIR.
Masses in the table are in 10^9 solar masses, the units of the Lelli table.

Runs for one case of paper_params.py (cgm or no_cgm, on the command line), after that case's fits are done.
Column 3 of the core densities file (upsilon factor) is ignored. The densities are fitted with Upsilon 0.5 / 0.7.

Dark mass inside R_200 has two parts:
  - the analytic integral over the added gas, with the dark mass cutoff
  - disk dark mass = grid dark mass inside R_max - analytic dark mass inside R_max, all treated as enclosed
The grid values need a model run of a few minutes, so they are cached in OUTPUT_DIR. The cache is
rebuilt when the core densities or the parameters change.
"""
import contextlib
import hashlib
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from cgm_integral import evaluate_dm_mass_integral
from paper_params import PAPER_PARAMS, core_densities_file, model_args, output_dir, working_dir

SPARC_DIR = Path(__file__).resolve().parent
PROPERTIES_FILE = SPARC_DIR / "SPARC_Lelli2016c.mrt.txt"
MASS_MODELS_FILE = SPARC_DIR / "MassModels_Lelli2016c.mrt"

UPSILON_DISK = 0.5
UPSILON_BULGE = 0.7
HE_FACTOR = 1.33
HUBBLE = 71.0                   # km/s/Mpc
COSMIC_BARYON_FRACTION = 0.157  # Omega_b / Omega_m, Planck 2018

G_SI = 6.67430e-11
G_KPC = 4.30091e-6              # kpc (km/s)^2 / Msun
C_SI = 2.9979e8
M_P = 1.67262e-27
KPC_TO_M = 3.086e19
M_SUN = 1.98847e30
MPC_TO_M = KPC_TO_M * 1000.0

RHO_CRIT = 3.0 * (HUBBLE * 1000.0 / MPC_TO_M) ** 2 / (8.0 * np.pi * G_SI)  # kg/m^3
TO_1E9_MSUN = 1.0 / (M_SUN * 1e9)
PER_CC_TO_M3 = 100.0 ** 3


def use_case(case):
    """Sets the parameters and paths below from paper_params.py for one case."""
    global CORE_DENSITIES_FILE, OUTPUT_DIR, GRID_CACHE_FILE, POWER_W, DM_CUTOFF_DENSITY, CORE_GAS_BETA, CORE_GAS_EXPONENT
    global R_CORE_FACTOR_GAS, CGM_CENTRAL_DENSITY, CGM_BETA, CGM_EXPONENT, R_CORE_FACTOR_CGM, GRID_NUM_MINUTES
    args = model_args(case)
    CORE_DENSITIES_FILE = core_densities_file(case)
    OUTPUT_DIR = Path(output_dir(f"added_gas_{case}"))
    GRID_CACHE_FILE = OUTPUT_DIR / "grid_dm_r_max.txt"
    POWER_W = PAPER_PARAMS["power_w"]
    DM_CUTOFF_DENSITY = args["dm_cutoff_density"]
    CORE_GAS_BETA = args["add_core_gas_beta"]
    CORE_GAS_EXPONENT = -CORE_GAS_BETA  # add_core_gas_beta is used directly as the exponent
    R_CORE_FACTOR_GAS = args["r_core_factor_gas"]
    CGM_CENTRAL_DENSITY = args["add_real_cgm_central_density"]  # particles per cc
    CGM_BETA = args["cgm_beta"]
    CGM_EXPONENT = -3.0 * CGM_BETA / 2.0
    R_CORE_FACTOR_CGM = args["r_core_factor_cgm"]
    GRID_NUM_MINUTES = PAPER_PARAMS["effort"]["table_num_minutes"]  # grid resolution of the model run


def load_core_densities(path):
    densities = {}
    for line in open(path):
        parts = line.split()
        if len(parts) >= 2:
            densities[parts[0]] = float(parts[1])
    return densities


def load_properties(path):
    """Returns {galaxy: (L[3.6] in 10^9 Lsun, MHI in 10^9 Msun, Vflat km/s)}."""
    properties = {}
    for line in open(path):
        parts = line.split()
        if len(parts) >= 18 and parts[1].lstrip('-').isdigit():
            properties[parts[0]] = (float(parts[7]), float(parts[13]), float(parts[15]))
    return properties


def load_last_points(path):
    """Returns {galaxy: (R kpc, Vobs km/s, Vbul km/s)} at the last observed radius."""
    last_points = {}
    for line in open(path):
        parts = line.split()
        if len(parts) == 10:
            try:
                last_points[parts[0]] = (float(parts[2]), float(parts[3]), float(parts[7]))
            except ValueError:
                pass
    return last_points


def load_grid_dark_mass(galaxy_names):
    """Returns {galaxy: model grid dark mass inside R_max in 10^9 Msun}, disk gas included. Runs the model if not cached."""
    params = (POWER_W, CORE_GAS_BETA, R_CORE_FACTOR_GAS, CGM_CENTRAL_DENSITY, CGM_BETA, R_CORE_FACTOR_CGM, DM_CUTOFF_DENSITY, GRID_NUM_MINUTES)
    cache_key = f"{params} {hashlib.sha256(CORE_DENSITIES_FILE.read_bytes()).hexdigest()}"

    if GRID_CACHE_FILE.exists():
        lines = GRID_CACHE_FILE.read_text().splitlines()
        if lines and lines[0] == f"# {cache_key}":
            print(f"Using cached grid dark mass from {GRID_CACHE_FILE}")
            return {line.split()[0]: float(line.split()[1]) for line in lines[1:]}

    print("Running the model for the grid dark mass, this takes a few minutes...")
    os.chdir(SPARC_DIR.parent)  # the model opens its data files relative to the repo root
    from sparc_dm_bruteCGM import run_sparc_dm_brute_cgm
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        _, _, _, galaxies = run_sparc_dm_brute_cgm(
            add_core_gas_central_density=0.0,  # ignored, the densities come from the core densities file
            add_core_gas_beta=CORE_GAS_BETA,
            r_core_factor_gas=R_CORE_FACTOR_GAS,
            add_real_cgm_central_density=CGM_CENTRAL_DENSITY,
            r_core_factor_cgm=R_CORE_FACTOR_CGM,
            cgm_beta=CGM_BETA,
            num_minutes=GRID_NUM_MINUTES,
            output_graphs=False,
            power_override=POWER_W,
            output_dir_override=working_dir(f"added_gas_grid_{case}"),
            use_core_gas_file=True,
            core_file=CORE_DENSITIES_FILE,
            only_galaxies=galaxy_names,
            dm_cutoff_density=DM_CUTOFF_DENSITY,
            return_galaxies=True,
        )
    grid_dark_mass = {name: POWER_W / C_SI ** 3 * g["sum_dm_mass_integral"] * TO_1E9_MSUN for name, g in galaxies.items()}

    with open(GRID_CACHE_FILE, "w") as f:
        f.write(f"# {cache_key}\n")
        for name, value in grid_dark_mass.items():
            f.write(f"{name} {value:.6e}\n")
    return grid_dark_mass


def integrate(r_kpc, gas_only, n0_core, rc_core_kpc, n0_cgm, rc_cgm_kpc):
    """Particles (gas_only) or dark mass in particle units, inside r_kpc. Densities in particles per cc."""
    value, _ = evaluate_dm_mass_integral(
        0, r_kpc * KPC_TO_M, gas_only, POWER_W, C_SI,
        rc_core_kpc * KPC_TO_M, n0_core * PER_CC_TO_M3, CORE_GAS_EXPONENT,
        rc_cgm_kpc * KPC_TO_M, n0_cgm * PER_CC_TO_M3, CGM_EXPONENT, DM_CUTOFF_DENSITY * PER_CC_TO_M3)
    return value


def gas_mass(r_kpc, n0, rc_kpc, exponent_is_core):
    """Gas mass inside r_kpc in 10^9 Msun, helium included, for one profile on its own."""
    if exponent_is_core:
        particles = integrate(r_kpc, True, n0, rc_kpc, 0.0, 1.0)
    else:
        particles = integrate(r_kpc, True, 0.0, 1.0, n0, rc_kpc)
    return particles * M_P * HE_FACTOR * TO_1E9_MSUN


def dark_mass(r_kpc, n0_core, rc_core_kpc, n0_cgm, rc_cgm_kpc):
    """Dark mass inside r_kpc in 10^9 Msun. The two profiles combine before the 2/3 power, so they can't be split."""
    return integrate(r_kpc, False, n0_core, rc_core_kpc, n0_cgm, rc_cgm_kpc) * M_P * TO_1E9_MSUN


def solve_r200(lelli_mass, disk_dark_mass, n0_core, rc_core_kpc, n0_cgm, rc_cgm_kpc):
    """Radius where the mean enclosed density is 200 rho_crit. Lelli mass and disk dark mass are treated as enclosed."""
    def excess(r_kpc):
        enclosed = (lelli_mass + disk_dark_mass
                    + gas_mass(r_kpc, n0_core, rc_core_kpc, True)
                    + gas_mass(r_kpc, n0_cgm, rc_cgm_kpc, False)
                    + dark_mass(r_kpc, n0_core, rc_core_kpc, n0_cgm, rc_cgm_kpc))
        return enclosed - (4.0 / 3.0) * np.pi * (r_kpc * KPC_TO_M) ** 3 * 200.0 * RHO_CRIT * TO_1E9_MSUN
    return brentq(excess, 0.01, 1e4, xtol=1e-3)


def build_row(name, n0_core, properties, last_points, grid_dark_mass):
    luminosity, hi, v_flat = properties[name]
    r_max, v_last, v_bul = last_points[name]

    # The bulge is spherical in Lelli and Vbul is for M/L = 1, so V^2 R / G at the last radius is its luminosity
    bulge_luminosity = min(v_bul ** 2 * r_max / G_KPC / 1e9, luminosity)
    disk_luminosity = luminosity - bulge_luminosity
    lelli_gas = HE_FACTOR * hi
    lelli_mass = UPSILON_DISK * disk_luminosity + UPSILON_BULGE * bulge_luminosity + lelli_gas

    # Same core radius as sparc_dm_bruteCGM.py, which uses L + HI rather than the Upsilon weighted mass
    r_core_kpc = 2.0 * ((luminosity + hi) * 1e9 / 1e11) ** (1.0 / 3.0)
    rc_core = r_core_kpc * R_CORE_FACTOR_GAS
    rc_cgm = r_core_kpc * R_CORE_FACTOR_CGM
    profiles = (n0_core, rc_core, CGM_CENTRAL_DENSITY, rc_cgm)

    # The grid can land slightly under the analytic value where the disk adds little, so floor at zero
    disk_dark_mass = max(grid_dark_mass[name] - dark_mass(r_max, *profiles), 0.0)

    r_200 = solve_r200(lelli_mass, disk_dark_mass, *profiles)
    core_r_max = gas_mass(r_max, n0_core, rc_core, True)
    core_r_200 = gas_mass(r_200, n0_core, rc_core, True)
    cgm_r_max = gas_mass(r_max, CGM_CENTRAL_DENSITY, rc_cgm, False)
    cgm_r_200 = gas_mass(r_200, CGM_CENTRAL_DENSITY, rc_cgm, False)
    added_r_max = core_r_max + cgm_r_max
    added_r_200 = core_r_200 + cgm_r_200
    dm_r_200 = dark_mass(r_200, *profiles) + disk_dark_mass

    return {
        "Galaxy": name,
        "n0_core": n0_core,
        "L36": luminosity,
        "HI": hi,
        "Gas": lelli_gas,
        "Mass": lelli_mass,
        "R_max": r_max,
        "R200": r_200,
        "R200_V": 100.0 * v_last / HUBBLE,
        "Core_Rmax": core_r_max,
        "Core_R200": core_r_200,
        "CGM_Rmax": cgm_r_max,
        "CGM_R200": cgm_r_200,
        "Add_Rmax": added_r_max,
        "Add_R200": added_r_200,
        "Add/Gas": added_r_max / lelli_gas if lelli_gas > 0 else np.nan,
        "Add/Mass": added_r_max / lelli_mass,
        "DM_Rmax": grid_dark_mass[name],
        "DM_Disk": disk_dark_mass,
        "DM_R200": dm_r_200,
        "Tot_R200": lelli_mass + added_r_200 + dm_r_200,
        "Vflat": v_flat,
    }


def write_table(rows, path):
    columns = list(rows[0].keys())
    with open(path, "w") as f:
        f.write(f"# Masses in 10^9 Msun, radii in kpc, n0_core in particles/cc. P = {POWER_W} W, H0 = {HUBBLE}, dark mass cutoff n_c = {DM_CUTOFF_DENSITY}/cc.\n")
        f.write("# Gas = 1.33 HI. Mass = 0.5 L_disk + 0.7 L_bulge + Gas. Added gas includes the 1.33 helium factor.\n")
        f.write("# R200: mean enclosed density of Tot_R200 = 200 rho_crit. R200_V: the old 100 V_last / H0 estimate.\n")
        f.write("# Add/Gas and Add/Mass use Add_Rmax. Vflat 0 = not measured.\n")
        f.write("# DM_Rmax: model grid dark mass inside R_max. DM_Disk: DM_Rmax minus the analytic added gas part.\n")
        f.write("# DM_R200: analytic added gas dark mass inside R200 + DM_Disk.\n")
        f.write(f"{columns[0]:<12}" + "".join(f"{c:>11}" for c in columns[1:]) + "\n")
        for row in rows:
            f.write(f"{row['Galaxy']:<12}" + "".join(f"{row[c]:>11.4g}" for c in columns[1:]) + "\n")


BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#52514e"
GRID = "#e4e3df"


def new_axes(xlabel, ylabel, title):
    fig, ax = plt.subplots(figsize=(7, 5.2), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.set_title(title, loc="left", fontsize=12)
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK, which="both", length=3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    return fig, ax


def points(ax, x, y, color=BLUE, label=None):
    ax.scatter(x, y, s=26, color=color, alpha=0.75, edgecolors="#fcfcfb", linewidths=0.5, label=label, zorder=3)


def ratio_lines(ax, x, y, ratios):
    """Dashed y = ratio * x reference lines across the data range, labelled at the right end."""
    x_max = np.nanmax(x)
    lo = np.nanmin(x)
    y_max = np.nanmax(y)
    for ratio in ratios:
        end = min(x_max, y_max / ratio)
        xs = np.array([lo, end])
        ax.plot(xs, ratio * xs, color=INK, linewidth=0.9, linestyle="--", zorder=2)
        ax.annotate(f"{ratio:g}:1", (end, ratio * end), xytext=(4, 0), textcoords="offset points", color=INK, fontsize=9)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / name, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


def column(rows, key):
    return np.array([row[key] for row in rows], dtype=float)


def make_graphs(rows):
    gas, mass = column(rows, "Gas"), column(rows, "Mass")
    added_r_max = column(rows, "Add_Rmax")
    added_r_200 = column(rows, "Add_R200")
    total = column(rows, "Tot_R200")
    r_200, r_200_v = column(rows, "R200"), column(rows, "R200_V")
    v_flat = column(rows, "Vflat")
    unit = r"$10^9\ M_\odot$"

    has_gas = gas > 0
    fig, ax = new_axes(f"Lelli gas mass, 1.33 HI ({unit})", f"Added gas inside $R_{{max}}$ ({unit})",
                       "Added gas vs observed gas")
    points(ax, gas[has_gas], added_r_max[has_gas])
    ratio_lines(ax, gas[has_gas], added_r_max[has_gas], [1, 0.1])
    save(fig, "1_added_gas_vs_lelli_gas.png")

    fig, ax = new_axes(f"Lelli total mass ({unit})", f"Added gas inside $R_{{max}}$ ({unit})",
                       "Added gas vs observed baryons")
    points(ax, mass, added_r_max)
    ratio_lines(ax, mass, added_r_max, [1, 0.1])
    save(fig, "2_added_gas_vs_lelli_mass.png")

    fig, ax = new_axes(f"Lelli total mass ({unit})", f"Total mass inside $R_{{200}}$ ({unit})",
                       "Halo mass vs observed baryons")
    points(ax, mass, total)
    ratio_lines(ax, mass, total, [100, 10, 1])
    save(fig, "3_total_r200_vs_lelli_mass.png")

    fig, ax = new_axes(f"Lelli total mass ({unit})", r"Baryon fraction inside $R_{200}$",
                       "Baryon fraction vs observed baryons")
    ax.set_yscale("linear")
    ax.set_ylim(0, 1)
    points(ax, mass, (mass + added_r_200) / total)
    ax.axhline(COSMIC_BARYON_FRACTION, color=INK, linewidth=0.9, linestyle="--", zorder=2)
    ax.annotate(f"cosmic mean {COSMIC_BARYON_FRACTION}", (ax.get_xlim()[0], COSMIC_BARYON_FRACTION),
                xytext=(6, 4), textcoords="offset points", color=INK, fontsize=9)
    save(fig, "4_baryon_fraction_vs_lelli_mass.png")

    # Baryonic Tully-Fisher: is the relation still tight once the added gas is counted?
    has_v_flat = v_flat > 0
    fig, ax = new_axes(r"$V_{flat}$ (km/s)", f"Baryonic mass ({unit})", "Baryonic Tully-Fisher, with and without added gas")
    for y, color, name in ((mass, BLUE, "Lelli mass"), (mass + added_r_max, ORANGE, r"Lelli + added gas inside $R_{max}$")):
        log_v, log_m = np.log10(v_flat[has_v_flat]), np.log10(y[has_v_flat])
        slope, intercept = np.polyfit(log_v, log_m, 1)
        scatter = np.std(log_m - (slope * log_v + intercept))
        print(f"BTFR {name}: slope {slope:.2f}, rms scatter {scatter:.3f} dex, {has_v_flat.sum()} galaxies")
        points(ax, v_flat[has_v_flat], y[has_v_flat], color=color, label=f"{name}: slope {slope:.2f}, scatter {scatter:.2f} dex")
    ax.legend(frameon=False, labelcolor=INK, fontsize=9)
    save(fig, "5_baryonic_tully_fisher.png")

    fig, ax = new_axes(r"Old estimate, $100\ V_{last} / H_0$ (kpc)", r"$R_{200}$ at 200 $\rho_{crit}$ (kpc)",
                       r"$R_{200}$: solved vs old estimate")
    points(ax, r_200_v, r_200)
    ratio_lines(ax, r_200_v, r_200, [1])
    save(fig, "6_r200_vs_old_estimate.png")


def main(case="cgm"):
    use_case(case)
    core_densities = load_core_densities(CORE_DENSITIES_FILE)
    properties = load_properties(PROPERTIES_FILE)
    last_points = load_last_points(MASS_MODELS_FILE)
    OUTPUT_DIR.mkdir(exist_ok=True)
    grid_dark_mass = load_grid_dark_mass(list(core_densities))
    rows = [build_row(name, n0, properties, last_points, grid_dark_mass) for name, n0 in core_densities.items()]

    table_file = OUTPUT_DIR / "added_gas_table.txt"
    write_table(rows, table_file)
    make_graphs(rows)

    inside = sum(row["R200"] < row["R_max"] for row in rows)
    print(f"{len(rows)} galaxies written to {table_file}")
    print(f"R200 inside R_max for {inside} galaxies (treating all Lelli mass as enclosed is then only approximate)")
    print(f"Total added gas inside R_max / total Lelli gas: {sum(r['Add_Rmax'] for r in rows) / sum(r['Gas'] for r in rows):.3f}")
    print(f"Total added gas inside R_max / total Lelli mass: {sum(r['Add_Rmax'] for r in rows) / sum(r['Mass'] for r in rows):.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "cgm")
