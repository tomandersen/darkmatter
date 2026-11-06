"""Global fit with one core gas central density and beta for all galaxies, run through paper.py.

run_free_power: power free, minimize chi2 and maximize power. The tradeoff data behind the choice of power,
core gas beta and central density in paper_params.py.
run_fixed_power: power fixed at the publication value, minimize chi2. A dense look at beta and density there.
Neither one feeds later steps: those use the values chosen in paper_params.py.
"""
import contextlib
import os
from pathlib import Path

import joblib
import optuna
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState

from paper_params import PAPER_PARAMS, cgm_args, fits_dir, path_suffix
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


def run_free_power(case):
    """Pareto front of chi2 against power. Read it from global_fits.db with optuna-dashboard or best_trials."""
    study = run_study(f"free_power_{case}", case, PAPER_PARAMS["effort"]["free_power_trials"], 0, ["minimize", "maximize"])
    for best_trial in sorted(study.best_trials, key=lambda t: t.values[1]):
        print(f"chi2/pt {best_trial.values[0]:.4f} P {best_trial.values[1]:.2f} W {best_trial.params}")
    return study


def run_fixed_power(case):
    """Best fits at the publication power, written to a table to compare with the chosen beta and density."""
    power = PAPER_PARAMS["power_w"]
    study = run_study(f"fixed_power_{case}_P{power}", case, PAPER_PARAMS["effort"]["fixed_power_trials"], power, ["minimize"])
    trials = sorted(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)), key=lambda t: t.value)
    table = f"# Single core gas density and beta at P = {power} W, case {case}, best chi2 first\n"
    table += f"{'chi2/pt':>9}{'beta':>8}{'density':>9}{'unobserved_gas':>16}\n"
    for trial in trials:
        table += (f"{trial.value:>9.4f}{trial.params['add_core_gas_beta']:>8.2f}{trial.params['add_core_gas_central_density']:>9.3f}"
                  f"{trial.user_attrs['unobserved_gas_fraction']:>16.3f}\n")
    (fits_dir() / f"fixed_power_{case}.txt").write_text(table)
    print("".join(table.splitlines(keepends=True)[:12]))
    return study


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent.parent)  # the model opens its data files relative to the repo root
    run_free_power("cgm")
