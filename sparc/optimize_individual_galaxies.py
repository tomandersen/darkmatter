import os
import sys
from pathlib import Path
import joblib
import optuna
from contextlib import redirect_stdout
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState


n_workers = 4


# Fit one core gas central density per galaxy, with everything else fixed from paper_params.py.
# Each galaxy is one study in the case's database. Pass the case (cgm or no_cgm) on the command line.
# Run the global fit for the case first (see paper.py): it sets the core gas beta.

# Ensure sparc directory is in sys.path
sys.path.insert(0, str(Path(__file__).parent))

# Ensure CWD is repository root so relative paths (e.g. ./sparc/...) in sparc_dm_bruteCGM resolve correctly
repo_root = Path(__file__).resolve().parent.parent
if (repo_root / "sparc").exists():
    os.chdir(repo_root)

from paper_params import PAPER_PARAMS, core_densities_file, model_args, run_dir, working_dir
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
                chi2_per_point, best_power, unobserved_gas_fraction = run_sparc_dm_brute_cgm(
                    add_core_gas_central_density=add_core_gas_central_density,
                    output_graphs=False,
                    upsilon_factor=upsilon_factor,
                    **fit_args
                )

        # Record chi squares per point and additional statistics
        trial.set_user_attr("chi2_per_point", float(chi2_per_point))
        trial.set_user_attr("power", float(best_power))
        trial.set_user_attr("unobserved_gas_fraction", float(unobserved_gas_fraction))

        return float(chi2_per_point)

    # No n_trials here: the callback enforces the limit across all workers. Failed trials
    # (eg a NaN chi2) count towards it, otherwise a galaxy that never completes would run forever.
    max_trials_callback = MaxTrialsCallback(n_trials_target, states=(TrialState.COMPLETE, TrialState.FAIL))
    study.optimize(objective, callbacks=[max_trials_callback])


def optimize_individual_galaxies(case):
    mrt_file = Path(__file__).parent / "MassModels_Lelli2016c.mrt"
    galaxy_names = PAPER_PARAMS["only_galaxies"] or load_galaxy_names(mrt_file)
    print(f"Fitting {len(galaxy_names)} galaxies for the {case} case")

    db_file = run_dir() / f"individual_{case}.db"
    storage_url = f"sqlite:///{db_file.resolve()}"
    print(f"Optuna database storage: {storage_url}")

    fixed_args = dict(
        model_args(case),
        num_minutes=PAPER_PARAMS["effort"]["individual_num_minutes"],
        power_override=PAPER_PARAMS["power_w"],
        min_mass=0,
        max_mass=1e99,
        output_dir_override=working_dir(f"individual_core_gas_{case}"),
    )

    n_trials_target = PAPER_PARAMS["effort"]["individual_trials"]

    core_densities = {}
    upsilon_factors = {}
    output_file = core_densities_file(case)
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
    optimize_individual_galaxies(sys.argv[1] if len(sys.argv) > 1 else "cgm")
