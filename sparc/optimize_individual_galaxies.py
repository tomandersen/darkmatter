import os
import sys
import time
from datetime import datetime
from pathlib import Path
import optuna
from contextlib import redirect_stdout

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


def write_core_densities_file(core_densities, filepath):
    """Write core densities to a two-column space-based text file."""
    with open(filepath, "w") as f:
        for gal_name, density in core_densities.items():
            f.write(f"{gal_name:<20} {density:6.3f}\n")


def optimize_individual_galaxies():
    mrt_file = Path(__file__).parent / "MassModels_Lelli2016c.mrt"
    galaxy_names = load_galaxy_names(mrt_file)
    print(f"Loaded {len(galaxy_names)} galaxies from {mrt_file.name}")

    # Database file: optimize_individual_YYYY_MM_DD.db
    date_str = datetime.now().strftime("%Y_%m_%d")
    db_file = Path(__file__).parent / f"optimize_individual_central_gas_cc_{date_str}.db"
    storage_url = f"sqlite:///{db_file.resolve()}"
    print(f"Optuna database storage: {storage_url}")

    # Fixed parameters from specifications
    add_core_gas_beta = 2.25
    r_core_factor_gas = 2.5
    add_real_cgm_central_density = 0.025
    r_core_factor_cgm = 0.20
    cgm_beta = 0.68
    num_minutes = 8
    output_graphs = False
    power_override = 16.0
    min_mass = 0
    max_mass = 1e99

    n_trials_target = 20

    core_densities = {}
    output_file = Path(__file__).parent / "core_densities.txt"
    if output_file.exists():
        with open(output_file, "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        core_densities[parts[0]] = float(parts[1])
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
                write_core_densities_file(core_densities, output_file)
            continue

        def objective(trial):
            # One variable to vary
            add_core_gas_central_density = trial.suggest_float("add_core_gas_central_density", 0.0, 2.5)
            # Redirect stdout to os.devnull (the system's "black hole")
            with open(os.devnull, 'w') as fnull:
                with redirect_stdout(fnull):
                    chi2_per_point, best_power, dm_to_hi_ratio = run_sparc_dm_brute_cgm(
                        add_core_gas_central_density=add_core_gas_central_density,
                        add_core_gas_beta=add_core_gas_beta,
                        r_core_factor_gas=r_core_factor_gas,
                        add_real_cgm_central_density=add_real_cgm_central_density,
                        r_core_factor_cgm=r_core_factor_cgm,
                        cgm_beta=cgm_beta,
                        num_minutes=num_minutes,
                        output_graphs=output_graphs,
                        power_override=power_override,
                        min_mass=min_mass,
                        max_mass=max_mass,
                        only_galaxies=[gal_name]
                    )

            # Record chi squares per point and additional statistics
            trial.set_user_attr("chi2_per_point", float(chi2_per_point))
            trial.set_user_attr("power", float(best_power))
            trial.set_user_attr("dm_to_hi_ratio", float(dm_to_hi_ratio))

            return float(chi2_per_point)

        study.optimize(objective, n_trials=remaining_trials)
        print(f"Completed {gal_name} -> Best chi2/pt: {study.best_value:.4f}, Best params: {study.best_params}")
        
        # Record best parameter in core_densities dict and write core_densities.txt
        core_densities[gal_name] = study.best_params["add_core_gas_central_density"]
        write_core_densities_file(core_densities, output_file)

        # ok - now use the best parameters and turn graphs ON
        run_sparc_dm_brute_cgm(
                add_core_gas_central_density=study.best_params["add_core_gas_central_density"],
                add_core_gas_beta=add_core_gas_beta,
                r_core_factor_gas=r_core_factor_gas,
                add_real_cgm_central_density=add_real_cgm_central_density,
                r_core_factor_cgm=r_core_factor_cgm,
                cgm_beta=cgm_beta,
                num_minutes=num_minutes,
                output_graphs=True,
                power_override=power_override,
                min_mass=min_mass,
                max_mass=max_mass,
                only_galaxies=[gal_name]
            )

    write_core_densities_file(core_densities, output_file)
    print(f"\nWrote {len(core_densities)} core densities to {output_file}")


if __name__ == "__main__":
    optimize_individual_galaxies()
