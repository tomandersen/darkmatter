import os
import sys
import time
from datetime import datetime
from pathlib import Path
import joblib
import optuna
from contextlib import redirect_stdout
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState


output_dir = "./sparc/dm_mass_individual_core_gas_prod"
n_workers = 4


# Optimize: Run through all the galaxies (load the galaxy names from the MassModels_Lelli2016c.mrt file)
# Keep the same function call parameters, and let core gas density (ie one variable vary)
# make a new database file called optimize_individual_YYYY_MM_DD.db 
# the study names will be galaxyname
# So there will be about 175 galaxies to fit, each one has a study. 
# I need to record the chi squares per point for each run in a study.
# Try 100 trials per galaxy. 
# vary
#     add_core_gas_central_density = trial.suggest_float("add_core_gas_central_density", 0.0, 2.0)
# Fix the power - 16.0
# add_core_gas_beta - 2.25
#         add_core_gas_beta=2.3, 
        # r_core_factor_gas=2.5, 
        # add_real_cgm_central_density=0.025,  # 0.025
        # r_core_factor_cgm=0.20, 
        # cgm_beta = 0.68,
        # num_minutes=8, 
        # output_graphs=False,
        # power_override = 16.0,
        # min_mass = 0,
        # max_mass = 1e99
# there is an argument to run_sparc_dm_brute_cgm which allows setting the galaxies to work on.

# Ensure sparc directory is in sys.path
sys.path.insert(0, str(Path(__file__).parent))

# Ensure CWD is repository root so relative paths (e.g. ./sparc/...) in sparc_dm_bruteCGM resolve correctly
repo_root = Path(__file__).resolve().parent.parent
if (repo_root / "sparc").exists():
    os.chdir(repo_root)

from sparc_dm_bruteCGM import run_sparc_dm_brute_cgm


def load_galaxy_names(mrt_path):
    """Load ordered unique galaxy names from MassModels_Lelli2016c.mrt."""
    names = []
    with open(mrt_path, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 10:
                try:
                    float(parts[1])
                    if parts[0] not in names:
                        names.append(parts[0])
                except ValueError:
                    pass
    return names


def write_core_densities_file(core_densities, upsilon_factors, filepath=None):
    """Write core densities and upsilon factors to a three-column space-based text file."""
    if filepath is None and isinstance(upsilon_factors, (str, Path)):
        filepath = upsilon_factors
        upsilon_factors = {}
    with open(filepath, "w") as f:
        for gal_name, density in core_densities.items():
            u_val = upsilon_factors.get(gal_name, 1.0)
            f.write(f"{gal_name:<20} {density:6.3f} {u_val:6.3f}\n")


def run_worker(gal_name, storage_url, n_trials_target, fit_args):
    """One worker process: all workers share the galaxy's study through the database."""
    study = optuna.load_study(study_name=gal_name, storage=storage_url)

    def objective(trial):
        # One variable to vary
        add_core_gas_central_density = trial.suggest_float("add_core_gas_central_density", 0.0, 6.0)
        upsilon_factor = 1.0  #trial.suggest_float("upsilon_factor", 0.6, 1.4) # allowing Upsilon factor can work for galaxies with inner maximal disks. (1 means mult both Upsilons by a factor of 1)
        # Redirect stdout to os.devnull (the system's "black hole")
        with open(os.devnull, 'w') as fnull:
            with redirect_stdout(fnull):
                chi2_per_point, best_power, dm_to_hi_ratio = run_sparc_dm_brute_cgm(
                    add_core_gas_central_density=add_core_gas_central_density,
                    output_graphs=False,
                    upsilon_factor=upsilon_factor,
                    **fit_args
                )

        # Record chi squares per point and additional statistics
        trial.set_user_attr("chi2_per_point", float(chi2_per_point))
        trial.set_user_attr("power", float(best_power))
        trial.set_user_attr("dm_to_hi_ratio", float(dm_to_hi_ratio))

        return float(chi2_per_point)

    # No n_trials here: the callback enforces the limit across all workers. Failed trials
    # (eg a NaN chi2) count towards it, otherwise a galaxy that never completes would run forever.
    max_trials_callback = MaxTrialsCallback(n_trials_target, states=(TrialState.COMPLETE, TrialState.FAIL))
    study.optimize(objective, callbacks=[max_trials_callback])


def optimize_individual_galaxies():
    mrt_file = Path(__file__).parent / "MassModels_Lelli2016c.mrt"
    galaxy_names = load_galaxy_names(mrt_file)
    print(f"Loaded {len(galaxy_names)} galaxies from {mrt_file.name}")



    # Database file: optimize_individual_YYYY_MM_DD.db
    date_str = datetime.now().strftime("%Y_%m_%d")
    db_file = Path(__file__).parent / f"optimize_individual_central_gas_{date_str}.db"
    storage_url = f"sqlite:///{db_file.resolve()}"
    print(f"Optuna database storage: {storage_url}")

    # Fixed parameters from specifications
    fixed_args = dict(
        add_core_gas_beta=2.25,
        r_core_factor_gas=2.5,
        add_real_cgm_central_density=0.025,
        r_core_factor_cgm=0.20,
        cgm_beta=0.68,
        num_minutes=10,
        power_override=16.0,
        min_mass=0,
        max_mass=1e99,
        output_dir_override=output_dir,
    )

    n_trials_target = 60

    core_densities = {}
    upsilon_factors = {}
    output_file = Path(__file__).parent / "core_densities_fresh.txt"
    if output_file.exists():
        with open(output_file, "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        core_densities[parts[0]] = float(parts[1])
                    except ValueError:
                        pass
                if len(parts) >= 3:
                    try:
                        upsilon_factors[parts[0]] = float(parts[2])
                    except ValueError:
                        pass

    for idx, gal_name in enumerate(galaxy_names, start=1):
        print(f"\n{'=' * 60}")
        print(f"[{idx}/{len(galaxy_names)}] Optimizing Galaxy: {gal_name}")
        print(f"{'=' * 60}")

        study = optuna.create_study(
            study_name=gal_name,
            storage=storage_url,
            load_if_exists=True,
            direction="minimize"
        )

        completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        remaining_trials = n_trials_target - completed_trials
        if remaining_trials <= 0:
            print(f"Galaxy '{gal_name}' already has {completed_trials} completed trials. Best chi2/pt: {study.best_value:.4f}")
            if study.best_params and "add_core_gas_central_density" in study.best_params:
                core_densities[gal_name] = study.best_params["add_core_gas_central_density"]
                if "upsilon_factor" in study.best_params:
                    upsilon_factors[gal_name] = study.best_params["upsilon_factor"]
                write_core_densities_file(core_densities, upsilon_factors, output_file)
            continue

        fit_args = dict(fixed_args, only_galaxies=[gal_name])
        joblib.Parallel(n_jobs=n_workers)(
            joblib.delayed(run_worker)(gal_name, storage_url, n_trials_target, fit_args)
            for _ in range(n_workers)
        )

        # Workers updated the study through the database, so reload it to see their trials
        study = optuna.load_study(study_name=gal_name, storage=storage_url)
        print(f"Completed {gal_name} -> Best chi2/pt: {study.best_value:.4f}, Best params: {study.best_params}")
        
        # Record best parameter in core_densities dict and write core_densities.txt
        best_upsilon_factor = 1.0
        core_densities[gal_name] = study.best_params["add_core_gas_central_density"]
        if "upsilon_factor" in study.best_params:
            upsilon_factors[gal_name] = study.best_params["upsilon_factor"]
            best_upsilon_factor  = study.best_params["upsilon_factor"]
        write_core_densities_file(core_densities, upsilon_factors, output_file)

        # ok - now use the best parameters and turn graphs ON
        run_sparc_dm_brute_cgm(
                add_core_gas_central_density=study.best_params["add_core_gas_central_density"],
                output_graphs=True,
                upsilon_factor=best_upsilon_factor,
                **fit_args
            )

    write_core_densities_file(core_densities, upsilon_factors, output_file)
    print(f"\nWrote {len(core_densities)} core densities and upsilon factors to {output_file}")


if __name__ == "__main__":
    optimize_individual_galaxies()
