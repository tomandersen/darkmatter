"""Runs everything for the paper, in order. Expect about 12 hours for the full run.

  1. Free power fit: the Pareto front of chi2 against power (with CGM). Stops for you to pick the power
  1b. Fixed power fit: core gas beta and density at that power (with CGM). Stops for you to pick both
  2. CGM sensitivity: why the CGM is set from observations and not fitted
  3. Individual galaxy fits: one core densities file for each case
  4. Tests: the graphs and numbers for the paper
  5. Added gas and halo tables, the only place R_200 is evaluated

All steps resume: stop it any time, and the same command picks up where it stopped. Numbers are in paper_params.py,
except the three picked from this run's results, which go in <run folder>/pareto_front.txt. After step 1 and step 1b
paper.py writes the tables to that file and asks for the numbers. Press return instead to stop, and fill them in the
file before running again. A number already in the file skips its step. A number given on the command line answers
its question, but its step still runs, so a long run can go start to end without stopping.
Everything goes to sparc/results/<run_tag>, with a copy of paper_params.py. Resuming stops if paper_params.py has changed:
change run_tag for a new run. --quick goes to sparc/results/quick, emptied each time.

  .venv/bin/python sparc/paper.py                    full run
  .venv/bin/python sparc/paper.py --power 16 --beta 3 --density 0.32   full run that answers its own questions
  .venv/bin/python sparc/paper.py --quick            a few minutes on 5 galaxies, to check that it all runs
  .venv/bin/python sparc/paper.py --fast             FAST_EFFORT, to try out settings. With --quick: the 5 galaxies at FAST_EFFORT
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

CASES = ("cgm", "no_cgm")
MOND_UPSILON_FILE = Path(__file__).resolve().parent / "mond_upsilon.txt"  # per-galaxy MOND Upsilon factors, from mond_upsilon_fit.py
QUICK_GALAXIES = ["DDO154", "NGC2403", "UGC02953", "NGC0289", "NGC7814"]
QUICK_EFFORT = dict(free_power_trials=4, fixed_power_trials=4, global_fit_num_minutes=0.3, individual_trials=4, individual_num_minutes=0.3,
                    sensitivity_num_minutes=0.1, test_num_minutes=0.3, newton_num_minutes=0.3, table_num_minutes=0.3)
FAST_EFFORT = dict(free_power_trials=200, fixed_power_trials=120, global_fit_num_minutes=1, individual_trials=60, individual_num_minutes=1,
                   sensitivity_num_minutes=1, test_num_minutes=3, newton_num_minutes=3, table_num_minutes=3)
DASHBOARD_FIRST_PORT = 8086
auto_choices = {}  # answers from the command line
FIXED_POWER_ROWS = 25  # best step 1b fits copied into pareto_front.txt


def timed_step(title, function, *args):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}", flush=True)
    start = time.time()
    result = function(*args)
    print(f"--- {title} took {(time.time() - start) / 3600:.2f} hours", flush=True)
    return result


def ask_choice(name, prompt):
    """Asks for one number, writes it to pareto_front.txt. Return on its own, or no terminal, stops paper.py."""
    from paper_params import choices_file, set_choice

    if name in auto_choices:
        print(f"{prompt}: {auto_choices[name]} (from the command line)")
        set_choice(name, auto_choices[name])
        return
    stop = f"Stopped. Set {name} in {choices_file()} and run paper.py again."
    if not sys.stdin.isatty():
        raise SystemExit(stop)
    while True:
        try:
            answer = input(f"{prompt} (return to stop): ").strip()
        except EOFError:  # ctrl-D
            answer = ""
        if not answer:
            raise SystemExit(stop)
        try:
            set_choice(name, float(answer))
            return
        except ValueError:
            print(f"Not a number: {answer}")


def pareto_table(study):
    table = f"{'chi2/pt':>9}{'P (W)':>9}{'beta':>8}{'density':>9}{'unobserved_gas':>16}\n"
    for trial in sorted(study.best_trials, key=lambda t: t.values[1]):
        table += (f"{trial.values[0]:>9.4f}{trial.values[1]:>9.2f}{trial.params['add_core_gas_beta']:>8.2f}"
                  f"{trial.params['add_core_gas_central_density']:>9.3f}{trial.user_attrs.get('unobserved_gas_fraction', float('nan')):>16.3f}\n")
    return table


def pick_power(run_free_power):
    from paper_params import add_choices_notes, choices_file, read_choices

    if read_choices()["power_w"] is not None and "power_w" not in auto_choices:
        print(f"\nStep 1 skipped: power_w = {read_choices()['power_w']} W, from {choices_file()}")
        return
    study = timed_step("Step 1: free power fit, with CGM", run_free_power, "cgm")
    add_choices_notes("Step 1: Pareto front of chi2 against power, with CGM", pareto_table(study))
    print(f"\nPareto front written to {choices_file()}")
    ask_choice("power_w", "Power in W")


def pick_core_gas(run_fixed_power):
    from paper_params import PAPER_PARAMS, add_choices_notes, choices_file, read_choices, run_dir

    choices = read_choices()
    names = ("core_gas_beta", "core_gas_central_density")
    if all(choices[name] is not None and name not in auto_choices for name in names):
        print(f"\nStep 1b skipped: core_gas_beta = {choices['core_gas_beta']}, "
              f"core_gas_central_density = {choices['core_gas_central_density']}, from {choices_file()}")
        return
    timed_step(f"Step 1b: fixed power fit at {PAPER_PARAMS['power_w']} W, with CGM", run_fixed_power, "cgm")
    lines = (run_dir() / "fixed_power_cgm.txt").read_text().splitlines()
    add_choices_notes(f"Step 1b: best fits at P = {PAPER_PARAMS['power_w']} W, with CGM. All of them in fixed_power_cgm.txt",
                      "\n".join(lines[1:FIXED_POWER_ROWS + 2]))
    print(f"Best fits written to {choices_file()}")
    if choices["core_gas_beta"] is None or "core_gas_beta" in auto_choices:
        ask_choice("core_gas_beta", "Core gas beta")
    if choices["core_gas_central_density"] is None or "core_gas_central_density" in auto_choices:
        ask_choice("core_gas_central_density", "Core gas central density, per cc")


def show_dashboards():
    """Makes the optuna databases now, so the dashboards can be opened before the fits start, and prints the commands."""
    import optuna
    from paper_params import FIT_DATABASES, fits_db_url

    print("\nWatch the fits with:")
    for port, name in enumerate(FIT_DATABASES, start=DASHBOARD_FIRST_PORT):
        url = fits_db_url(name)
        optuna.storages.RDBStorage(url)  # creates the file and its tables
        print(f"  optuna-dashboard --port {port} {url}")


def run_tests():
    from paper_params import PAPER_PARAMS, case_params, core_densities_file, model_args, output_dir, run_dir
    from sparc_dm_bruteCGM import run_sparc_dm_brute_cgm

    effort = PAPER_PARAMS["effort"]
    power = PAPER_PARAMS["power_w"]
    results_file = run_dir() / "test_results.json"
    results = json.loads(results_file.read_text()) if results_file.exists() else {}

    def run_test(name, folder, **kwargs):
        if name in results:
            print(f"Test {name} already done")
            return
        kwargs.setdefault("num_minutes", effort["test_num_minutes"])
        err_per_point, best_power, unobserved_gas_fraction = run_sparc_dm_brute_cgm(
            output_graphs=True, output_dir_override=output_dir(folder), **kwargs)
        results[name] = {"err_per_point": err_per_point, "power_w": best_power, "unobserved_gas_fraction": unobserved_gas_fraction}
        results_file.write_text(json.dumps(results, indent=2) + "\n")

    for case in CASES:
        # the core densities made with optimize_individual_galaxies.py
        run_test(f"core_file_{case}", f"core_file_{case}", add_core_gas_central_density=0.0,  # ignored as we are using the file
                 use_core_gas_file=True, core_file=core_densities_file(case), mond_upsilon_file=MOND_UPSILON_FILE, power_override=power, **model_args(case))
        # one core gas density for all galaxies, from paper_params.py
        run_test(f"single_gas_{case}", f"single_gas_{case}", add_core_gas_central_density=case_params(case)["core_gas_central_density"],
                 power_override=power, **model_args(case))

    no_added_gas = dict(model_args("no_cgm"), add_core_gas_central_density=0.0)  # the no_cgm case adds no CGM, and no core gas here
    # no gas added, and let global P vary
    run_test("no_gas", "no_gas", power_override=0.0, **no_added_gas)
    # a pure Newton run: set power to 1e-5, so it is effectively off
    run_test("newton", "Newton", power_override=0.00001, num_minutes=effort["newton_num_minutes"], **no_added_gas)

    print(f"\n{'Test':<20}{'err/pt':>10}{'P (W)':>12}{'unobserved gas':>16}")
    for name, row in results.items():
        print(f"{name:<20}{row['err_per_point']:>10.4f}{row['power_w']:>12.4g}{row['unobserved_gas_fraction']:>16.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="a few minutes on 5 galaxies, in sparc/results/quick, emptied first")
    parser.add_argument("--fast", action="store_true", help="FAST_EFFORT trials and minutes, to try out settings")
    parser.add_argument("--power", type=float, help="answer for the power in W, after step 1")
    parser.add_argument("--beta", type=float, help="answer for the core gas beta, after step 1b")
    parser.add_argument("--density", type=float, help="answer for the core gas central density per cc, after step 1b")
    args = parser.parse_args()

    os.chdir(Path(__file__).resolve().parent.parent)  # the model opens its data files relative to the repo root

    # Imported here so the worker processes, which import this file, don't load everything twice
    from paper_params import PAPER_PARAMS, case_params, start_run

    if args.quick:
        PAPER_PARAMS.update(run_tag="quick", only_galaxies=QUICK_GALAXIES)
        PAPER_PARAMS["effort"].update(QUICK_EFFORT)
    if args.fast:  # after --quick, so both together is the quick galaxies at FAST_EFFORT
        PAPER_PARAMS["effort"].update(FAST_EFFORT)
    run_folder = start_run(fresh=args.quick)
    answers = dict(power_w=args.power, core_gas_beta=args.beta, core_gas_central_density=args.density)
    auto_choices.update({name: value for name, value in answers.items() if value is not None})
    print(f"Results in {run_folder}")
    print(f"Effort: {PAPER_PARAMS['effort']}")
    show_dashboards()

    import added_gas_table
    from cgm_sensitivity import run_cgm_sensitivity
    from optimize_individual_galaxies import optimize_individual_galaxies
    from optimizer import run_fixed_power, run_free_power

    start = time.time()
    pick_power(run_free_power)
    pick_core_gas(run_fixed_power)
    timed_step("Step 2: CGM sensitivity", run_cgm_sensitivity)
    for case in CASES:
        timed_step(f"Step 3: individual galaxy fits, {case}", optimize_individual_galaxies, case)
    timed_step("Step 4: tests", run_tests)
    for case in CASES:
        timed_step(f"Step 5: added gas table, {case}", added_gas_table.main, case)

    print(f"\nAll done in {(time.time() - start) / 3600:.2f} hours. Results in {run_folder}")
    for case in CASES:
        print(f"{case}: {case_params(case)}")


# The guard matters: joblib workers import this file, and without it each one would start the whole pipeline
if __name__ == "__main__":
    main()
