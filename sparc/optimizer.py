import joblib
import optuna
import time
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
from sparc_dm_bruteCGM import run_sparc_dm_brute_cgm
from pathlib import Path

study_name = "dark_mass_opt_" + Path(__file__).resolve().parent.parent.name + "TPE-CHI-CGM-CUT"
storage = "sqlite:///optimization.db"
n_workers = 4
n_trials_target = 1000  # total across all workers, including trials already in the study


# 1. Define your objective function
def objective(trial):
    # Suggest 6 floating-point variables with predefined bounds
    add_core_gas_central_density = trial.suggest_float("add_core_gas_central_density", 0.2, 1.0)
    add_core_gas_beta = trial.suggest_float("add_core_gas_beta", 1.7, 3.0)
    r_core_factor_gas = 2.5# trial.suggest_float("r_core_factor_gas", 1, 5)
    add_real_cgm_central_density = 2.43e-4 #trial.suggest_float("add_real_cgm_central_density", 0,  0.1)
    r_core_factor_cgm = 6.0 #trial.suggest_float("r_core_factor_cgm", 0.1, 0.5)
    cgm_beta = 0.5 #trial.suggest_float("cgm_beta", 0.5, 1.0)

    num_minutes = 4
    output_graphs = False

    # Run your 10-minute code
    # power override - we set powe in all these optimiztions
    power_override = 0
    min_mass = 0
    max_mass = 1e99

    result, power, unobserved_gas_fraction = run_sparc_dm_brute_cgm(add_core_gas_central_density, add_core_gas_beta, r_core_factor_gas, add_real_cgm_central_density, r_core_factor_cgm, cgm_beta, num_minutes, output_graphs, power_override, min_mass, max_mass)
    trial.set_user_attr("unobserved_gas_fraction", unobserved_gas_fraction)

    return result, power # Optuna will try to minimize or maximize this


def make_sampler():
    return optuna.samplers.TPESampler(multivariate=True)


# 2. One worker process: all workers share the study through the database
def run_worker():
    # load_study does not remember the sampler - without this a multi-objective study gets NSGA-II
    study = optuna.load_study(study_name=study_name, storage=storage, sampler=make_sampler())
    # No n_trials here: the callback enforces the limit across all workers. Failed trials
    # (eg a NaN chi2) count towards it, otherwise a run that never completes would go on forever.
    max_trials_callback = MaxTrialsCallback(n_trials_target, states=(TrialState.COMPLETE, TrialState.FAIL))
    study.optimize(objective, callbacks=[max_trials_callback])


if __name__ == "__main__":
    # 3. Use a persistent SQLite database so you can stop/resume anytime
    optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        sampler=make_sampler(),
        directions=["minimize", "maximize"]  #  direction="minimize" # or "maximize"
    )

    # 4. Optimize with parallel workers
    joblib.Parallel(n_jobs=n_workers)(joblib.delayed(run_worker)() for _ in range(n_workers))

    # Two objectives, so there is a set of best trials (the Pareto front) rather than one best_params
    study = optuna.load_study(study_name=study_name, storage=storage)
    for best_trial in study.best_trials:
        print("Best values:", best_trial.values, "variables:", best_trial.params)
