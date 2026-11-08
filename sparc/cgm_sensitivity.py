"""How much does the fit inside R_max depend on the CGM? This is why the CGM is set from observations, not fitted.

Refits the core density of every galaxy for each CGM below, at the cgm case's core gas beta and the publication power.
Low resolution: for comparing the rows with each other. Finished rows are kept, so a rerun picks up where it stopped.
"""
import contextlib
import json
import os

import joblib
import numpy as np
from scipy.optimize import minimize_scalar

from paper_params import PAPER_PARAMS, model_args, paper_galaxies, run_dir, working_dir
from sparc_data import load_rotation_curves
from sparc_dm_bruteCGM import run_sparc_dm_brute_cgm

n_workers = 4
CORE_DENSITY_GRID = [0.0, 0.05, 0.2, 0.5, 1.0, 2.0, 4.0]

# name: central density per cc, core radius factor, beta
CGM_VARIANTS = {
    "publication": (2.43e-4, 6.0, 0.5),
    "none": (0.0, 6.0, 0.5),
    "0.3x_density": (7.29e-5, 6.0, 0.5),
    "3x_density": (7.29e-4, 6.0, 0.5),
    "smaller_core": (1.25e-3, 2.0, 0.5),    # same density at 50 r_core as publication
    "steeper": (7.67e-4, 6.0, 0.68),        # same density at 50 r_core as publication
    "previous": (0.025, 0.20, 0.68),        # the CGM used up to November 2026
}


def fit_galaxy(galaxy, args):
    def chi2(core_density):
        with contextlib.redirect_stdout(open(os.devnull, "w")):
            return run_sparc_dm_brute_cgm(add_core_gas_central_density=float(core_density), output_graphs=False, only_galaxies=[galaxy], **args)[0]
    values = [chi2(density) for density in CORE_DENSITY_GRID]
    best = int(np.argmin(values))
    low, high = CORE_DENSITY_GRID[max(best - 1, 0)], CORE_DENSITY_GRID[min(best + 1, len(CORE_DENSITY_GRID) - 1)]
    refined = minimize_scalar(chi2, bounds=(low, high), method="bounded", options={"maxiter": 4, "xatol": 0.01})
    if refined.fun < values[best]:
        return galaxy, float(refined.x), float(refined.fun)
    return galaxy, CORE_DENSITY_GRID[best], float(values[best])


def run_cgm_sensitivity():
    results_file = run_dir() / "cgm_sensitivity.jsonl"
    results = {}
    if results_file.exists():
        results = {row["name"]: row for row in map(json.loads, results_file.read_text().splitlines())}

    galaxies = paper_galaxies()
    points = {name: len(curve["R"]) for name, curve in load_rotation_curves(galaxies).items()}
    base_args = model_args("cgm")
    base_args.pop("only_galaxies")
    base_args.update(num_minutes=PAPER_PARAMS["effort"]["sensitivity_num_minutes"], power_override=PAPER_PARAMS["power_w"],
                     output_dir_override=working_dir("cgm_sensitivity"))

    with joblib.Parallel(n_jobs=n_workers) as pool:
        for name, (density, factor, beta) in CGM_VARIANTS.items():
            if name in results:
                continue
            args = dict(base_args, add_real_cgm_central_density=density, r_core_factor_cgm=factor, cgm_beta=beta)
            fits = pool(joblib.delayed(fit_galaxy)(galaxy, args) for galaxy in galaxies)
            total_points = sum(points[galaxy] for galaxy, _, _ in fits)
            results[name] = {"name": name, "cgm_central_density": density, "r_core_factor_cgm": factor, "cgm_beta": beta,
                             "chi2_per_point": sum(chi2 * points[galaxy] for galaxy, _, chi2 in fits) / total_points,
                             "median_chi2": float(np.median([chi2 for _, _, chi2 in fits])),
                             "mean_core_density": float(np.mean([core for _, core, _ in fits]))}
            with open(results_file, "a") as f:
                f.write(json.dumps(results[name]) + "\n")

    table_file = run_dir() / "cgm_sensitivity.txt"
    with open(table_file, "w") as f:
        f.write(f"# Core density refit per galaxy at P = {PAPER_PARAMS['power_w']} W. Low resolution: compare rows with each other.\n")
        f.write(f"{'CGM':<14}{'n0(/cc)':>10}{'factor':>8}{'beta':>7}{'chi2/pt':>9}{'median':>9}{'mean_n_core':>13}\n")
        for row in results.values():
            f.write(f"{row['name']:<14}{row['cgm_central_density']:>10.3g}{row['r_core_factor_cgm']:>8.2f}{row['cgm_beta']:>7.2f}"
                    f"{row['chi2_per_point']:>9.4f}{row['median_chi2']:>9.4f}{row['mean_core_density']:>13.3f}\n")
    print(table_file.read_text())
