"""Fit one Upsilon factor per galaxy for simple MOND, so MOND gets the same one-parameter-per-galaxy freedom as the dark mass fits.

MOND here uses only the Lelli 2016 mass models (Vgas, Vdisk, Vbul), so it needs none of the 3D work in sparc_dm_bruteCGM.py,
and the result is the same for the cgm and no_cgm cases. Each evaluation is microseconds, so a dense grid scan replaces optuna.

The physics and the chi^2 match sparc_dm_bruteCGM.py:
  Upsilon_disk = 0.5*f, Upsilon_bulge = 0.7*f (f is the same upsilon_factor convention)
  simple MOND: g = (g_N + sqrt(g_N^2 + 4 g_N a0)) / 2, a0 = 1.2e-10
  chi err = sqrt((0.15*V_mond)^2 + eVobs^2), eVobs <= 0 set to 1

f has a lognormal prior as in Li, Lelli et al. 2018: the fit minimises chi^2 + (log10 f / PRIOR_DEX)^2, with PRIOR_DEX = 0.1.
The reported chi2/pt is the data chi^2 alone, so it compares directly with the dark mass fits.

Writes mond_upsilon.txt next to this file: galaxy, best f, chi2/pt at best f, chi2/pt at f=1, number of points, and a flag if f hit a bound.
Usage: python sparc/mond_upsilon_fit.py [prior_dex [f_min f_max]]   (prior_dex 0 turns the prior off)
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
KINEMATICS_FILE = HERE / "MassModels_Lelli2016_SigmaGas.mrt"
OUTPUT_FILE = HERE / "mond_upsilon.txt"

UPSILON_DISK = 0.5
UPSILON_BULGE = 0.7
A0_MOND = 1.2e-10
CHI_ERR_V_TOT = 0.15
KPC_TO_M = 3.086e19

F_MIN, F_MAX, F_STEP = 0.1, 5.0, 0.002
PRIOR_DEX = 0.1  # width of the lognormal prior on f, in dex


def load_galaxies(path=KINEMATICS_FILE):
    """Rotation curves from the Lelli 2016 mass models, parsed as in sparc_dm_bruteCGM.py. Vdisk and Vbul are at Upsilon = 1."""
    galaxies = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) != 11 or line.startswith('---') or line.startswith('Byte'):
                continue
            try:
                R, Vobs, eVobs, Vgas, Vdisk, Vbul = (float(p) for p in parts[2:8])
            except ValueError:
                continue
            if eVobs <= 0:
                eVobs = 1.0
            g = galaxies.setdefault(parts[0], {k: [] for k in ('R', 'Vobs', 'eVobs', 'Vgas', 'Vdisk', 'Vbul')})
            for k, v in zip(('R', 'Vobs', 'eVobs', 'Vgas', 'Vdisk', 'Vbul'), (R, Vobs, eVobs, Vgas, Vdisk, Vbul)):
                g[k].append(v)
    return {name: {k: np.array(v) for k, v in g.items()} for name, g in galaxies.items()}


def mond_velocity(g, f):
    """Simple MOND rotation curve [km/s] with Upsilon_disk = 0.5 f, Upsilon_bulge = 0.7 f. f may be an array (returns len(f) x n_points)."""
    f = np.atleast_1d(f)[:, None]
    V_bar_sq = np.clip(f * (UPSILON_DISK * g['Vdisk']**2 + UPSILON_BULGE * g['Vbul']**2) + g['Vgas'] * np.abs(g['Vgas']), 0, None) * 1e6
    R_m = g['R'] * KPC_TO_M
    g_N = np.divide(V_bar_sq, R_m, out=np.zeros_like(V_bar_sq), where=R_m > 0)
    g_mond = (g_N + np.sqrt(g_N**2 + 4 * g_N * A0_MOND)) / 2.0
    return np.sqrt(g_mond * R_m) / 1000.0


def chi2(g, f):
    V = mond_velocity(g, f)
    err = np.sqrt((V * CHI_ERR_V_TOT)**2 + g['eVobs']**2)
    return np.sum(((g['Vobs'] - V) / err)**2, axis=1)


def prior_penalty(f, prior_dex):
    return (np.log10(f) / prior_dex)**2 if prior_dex > 0 else np.zeros_like(f)


def fit_galaxy(g, f_grid, prior_dex):
    c = chi2(g, f_grid)
    i = int(np.argmin(c + prior_penalty(f_grid, prior_dex)))
    return f_grid[i], c[i], chi2(g, 1.0)[0]


def main():
    prior_dex = float(sys.argv[1]) if len(sys.argv) >= 2 else PRIOR_DEX
    f_min, f_max = (float(sys.argv[2]), float(sys.argv[3])) if len(sys.argv) >= 4 else (F_MIN, F_MAX)
    f_grid = np.arange(f_min, f_max + F_STEP / 2, F_STEP)
    galaxies = load_galaxies()

    rows = []
    for name, g in galaxies.items():
        f_best, c_best, c_one = fit_galaxy(g, f_grid, prior_dex)
        at_bound = "low" if f_best <= f_min + F_STEP / 2 else "high" if f_best >= f_max - F_STEP / 2 else ""
        rows.append((name, f_best, c_best, c_one, len(g['R']), at_bound))

    n_total = sum(r[4] for r in rows)
    chi_best = sum(r[2] for r in rows)
    chi_one = sum(r[3] for r in rows)
    fs = np.array([r[1] for r in rows])
    summary = [
        f"{len(rows)} galaxies, {n_total} points, f grid [{f_min}, {f_max}], prior {prior_dex} dex",
        f"MOND chi^2 at f=1:          {chi_one:9.2f}  per point {chi_one / n_total:.3f}",
        f"MOND chi^2 at per-galaxy f: {chi_best:9.2f}  per point {chi_best / n_total:.3f}",
        f"f: median {np.median(fs):.3f}, 16-84% {np.percentile(fs, 16):.3f}-{np.percentile(fs, 84):.3f}",
        f"at low bound: {sum(r[5] == 'low' for r in rows)}, at high bound: {sum(r[5] == 'high' for r in rows)}",
    ]

    with open(OUTPUT_FILE, "w") as out:
        out.write(f"# Per-galaxy Upsilon factor f for simple MOND (a0={A0_MOND}), Upsilon_disk={UPSILON_DISK}*f, Upsilon_bulge={UPSILON_BULGE}*f\n")
        prior_text = f"lognormal prior on f, {prior_dex} dex" if prior_dex > 0 else "no prior on f"
        out.write(f"# {prior_text}; chi2pt is the data chi^2 only (no prior term)\n")
        out.write(f"# grid f in [{f_min}, {f_max}] step {F_STEP}; chi err = sqrt(({CHI_ERR_V_TOT}*V)^2 + eVobs^2). Made by mond_upsilon_fit.py\n")
        out.write("#\n# Summary (totals over all galaxies, per point = chi^2 total / number of points)\n")
        for line in summary:
            out.write(f"#   {line}\n")
        out.write("#\n")
        out.write(f"# {'galaxy':<18} {'f':>6} {'chi2pt':>8} {'chi2pt_f1':>10} {'n':>4}  bound\n")
        for name, f_best, c_best, c_one, n, at_bound in rows:
            out.write(f"{name:<20} {f_best:6.3f} {c_best / n:8.3f} {c_one / n:10.3f} {n:4d}  {at_bound}\n")

    for line in summary:
        print(line)
    print(f"Wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
