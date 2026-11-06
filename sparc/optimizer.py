"""Global fits with one core gas central density and beta for all galaxies, run through paper.py.

run_global_fit: power fixed, minimize chi2. Gives the core gas beta and density for a case.
run_free_power: power free, minimize chi2 and maximize power. The tradeoff data behind the choice of power.
"""
import contextlib
import os
from pathlib import Path

import joblib
import optuna
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState

from paper_params import PAPER_PARAMS, cgm_args, fits_dir, path_suffix, save_global_fit
from sparc_dm_bruteCGM import run_sparc_dm_brute_cgm

n_workers = 4
DENSITY_RANGE = (0.05, 1.5)
BETA_RANGE = (1.5, 6.0)  # used directly as the exponent. A top of 3.0 was limiting the fit near 24.7 W


def objective(trial, fit_args, power_override):
    density = trial.suggest_float("add_core_gas_central_density", *DENSITY_RANGE)
    beta = trial.suggest_float("add_core_gas_beta", *BETA_RANGE)
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        result, power, unobserved_gas_fraction = run_sparc_dm_brute_cgm(
            add_core_gas_central_density=density, add_core_gas_beta=beta, output_graphs=False, power_override=power_override, **fit_args)
    trial.set_user_attr("unobserved_gas_fraction", unobserved_gas_fraction)
    return result if power_override != 0 else (result, power)


def make_sampler():
    return optuna.samplers.TPESampler(multivariate=True)


def run_worker(study_name, storage, n_trials_target, fit_args, power_override):
    """One worker process: all workers share the study through the database."""
    # load_study does not remember the sampler - without this a multi-objective study gets NSGA-II
    study = optuna.load_study(study_name=study_name, storage=storage, sampler=make_sampler())
    # No n_trials here: the callback enforces the limit across all workers. Failed trials
    # (eg a NaN chi2) count towards it, otherwise a run that never completes would go on forever.
    max_trials_callback = MaxTrialsCallback(n_trials_target, states=(TrialState.COMPLETE, TrialState.FAIL))
    study.optimize(lambda trial: objective(trial, fit_args, power_override), callbacks=[max_trials_callback])


def run_study(study_name, case, n_trials_target, power_override, directions):
    """Runs the study up to n_trials_target, or just loads it if it is already there."""
    storage = f"sqlite:///{(fits_dir() / 'global_fits.db').resolve()}"
    study = optuna.create_study(study_name=study_name, storage=storage, load_if_exists=True, sampler=make_sampler(), directions=directions)
    done = len(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE, TrialState.FAIL)))
    if done < n_trials_target:
        fit_args = dict(cgm_args(case), num_minutes=PAPER_PARAMS["effort"]["global_fit_num_minutes"],
                        output_dir_override=f"./sparc/dm_mass_global_fit_{case}{path_suffix()}")
        joblib.Parallel(n_jobs=n_workers)(
            joblib.delayed(run_worker)(study_name, storage, n_trials_target, fit_args, power_override) for _ in range(n_workers))
    return optuna.load_study(study_name=study_name, storage=storage)


def at_range_edge(value, value_range):
    margin = 0.02 * (value_range[1] - value_range[0])
    return value < value_range[0] + margin or value > value_range[1] - margin


def run_global_fit(case):
    """Best single core gas density and beta at the publication power. Saved for the later steps."""
    power = PAPER_PARAMS["power_w"]
    study = run_study(f"global_fit_{case}_P{power}", case, PAPER_PARAMS["effort"]["global_fit_trials"], power, ["minimize"])
    density = study.best_params["add_core_gas_central_density"]
    beta = study.best_params["add_core_gas_beta"]
    fit = {"core_gas_beta": beta, "core_gas_central_density": density, "global_chi2_per_point": study.best_value,
           "n_trials": len(study.trials), "at_search_limit": at_range_edge(density, DENSITY_RANGE) or at_range_edge(beta, BETA_RANGE)}
    save_global_fit(case, fit)
    print(f"Global fit {case}: {fit}")
    if fit["at_search_limit"]:
        print(f"WARNING: the {case} global fit is at the edge of its search range, widen DENSITY_RANGE or BETA_RANGE in optimizer.py")
    return fit


def run_free_power(case):
    """Pareto front of chi2 against power. Read it from global_fits.db with optuna-dashboard or best_trials."""
    study = run_study(f"free_power_{case}", case, PAPER_PARAMS["effort"]["free_power_trials"], 0, ["minimize", "maximize"])
    for best_trial in sorted(study.best_trials, key=lambda t: t.values[1]):
        print(f"chi2/pt {best_trial.values[0]:.4f} P {best_trial.values[1]:.2f} W {best_trial.params}")
    return study


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent.parent)  # the model opens its data files relative to the repo root
    run_free_power("cgm")
